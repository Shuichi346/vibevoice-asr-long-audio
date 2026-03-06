#!/usr/bin/env python3
"""
VibeVoice-ASR 長時間音声対応スクリプト
Mac Mini 25GB Unified Memory + 4bit モデル用

核心: encode_speech() をチャンク化して、メモリスパイクを回避する
"""

import gc
import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_audio.stt.utils import load_model


# ================================================================
# 定数
# ================================================================
MODEL_ID = "mlx-community/VibeVoice-ASR-8bit"
AUDIO_PATH = "path/to/Folder/example.wav"
OUTPUT_PATH = "path/to/Folder"

# ここはまあいじらなくて良い
SAMPLE_RATE = 24000
HOP_LENGTH = 3200
PREFILL_STEP_SIZE = 512
MAX_TOKENS = 32768

# エンコーダチャンクの自動決定で使う基準値
# 24GB環境で 600秒が適正だった実測に基づく
MEMORY_REFERENCE_AVAILABLE_GB = 14.0  # モデルロード後の空きメモリ基準値(GB)
CHUNK_REFERENCE_SECONDS = 600         # 上記基準値で安全に動作したチャンク秒数
CHUNK_SAFETY_MARGIN = 0.80            # 安全マージン（80%に抑える）
CHUNK_MIN_SECONDS = 120               # 最小チャンクサイズ（2分）
CHUNK_MAX_SECONDS = 1800              # 最大チャンクサイズ（30分）


# ================================================================
# メモリベースのチャンクサイズ自動決定
# ================================================================

def get_available_memory_gb() -> float:
    """Metal GPUの空きメモリをGB単位で返す"""
    device_info = mx.metal.device_info()
    max_recommended = device_info["max_recommended_working_set_size"]
    active = mx.metal.get_active_memory()
    available = max_recommended - active
    return available / (1024 ** 3)


def calculate_encoder_chunk_seconds(available_gb: Optional[float] = None) -> int:
    """
    空きメモリに基づいてエンコーダのチャンクサイズ（秒）を決定する。

    実測データから線形比例で推定:
      24GB環境でモデルロード後の空き約14GBのとき、600秒が適正。
      エンコーダの中間バッファは音声長に概ね比例するため、
      空きメモリとの線形スケーリングで推奨値を算出する。
    """
    if available_gb is None:
        available_gb = get_available_memory_gb()

    # 基準値との比率でスケーリングし、安全マージンを適用
    ratio = available_gb / MEMORY_REFERENCE_AVAILABLE_GB
    raw_seconds = CHUNK_REFERENCE_SECONDS * ratio * CHUNK_SAFETY_MARGIN

    # HOP_LENGTH の倍数（秒換算）に切り捨て
    hop_seconds = HOP_LENGTH / SAMPLE_RATE
    aligned_seconds = int(raw_seconds / hop_seconds) * hop_seconds
    aligned_seconds = int(aligned_seconds)

    # 上限・下限でクランプ
    chunk_seconds = max(CHUNK_MIN_SECONDS, min(CHUNK_MAX_SECONDS, aligned_seconds))

    return chunk_seconds


# ================================================================
# 音声ロード
# ================================================================

def load_and_preprocess_audio(model, audio_path: str) -> mx.array:
    """音声をロード・前処理（リサンプリング・正規化）"""
    print(f"音声ファイル読み込み中: {audio_path}")
    audio_tensor = model._preprocess_audio(audio_path)
    duration = audio_tensor.shape[1] / SAMPLE_RATE
    print(f"音声長: {duration:.1f}秒 ({duration / 60:.1f}分)")
    print(f"サンプル数: {audio_tensor.shape[1]:,}")
    return audio_tensor


# ================================================================
# チャンクエンコード
# ================================================================

def chunked_encode_speech(
    model,
    audio_tensor: mx.array,
    chunk_seconds: int = 600,
    verbose: bool = True,
) -> mx.array:
    """
    音声エンコーダをチャンク単位で実行し、メモリスパイクを防ぐ。

    各チャンクのエンコード後にキャッシュクリアし、
    最終的にすべての特徴量を結合する。
    """
    total_samples = audio_tensor.shape[1]
    chunk_samples = chunk_seconds * SAMPLE_RATE

    # HOP_LENGTH の倍数に調整
    chunk_samples = (chunk_samples // HOP_LENGTH) * HOP_LENGTH

    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    print(f"\n=== チャンクエンコード ===")
    print(f"チャンクサイズ: {chunk_seconds}秒 ({chunk_samples:,}サンプル)")
    print(f"チャンク数: {num_chunks}")

    acoustic_features_list = []
    semantic_features_list = []

    for i in range(num_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, total_samples)
        chunk = audio_tensor[:, start:end]

        if verbose:
            chunk_dur = (end - start) / SAMPLE_RATE
            print(
                f"\n--- チャンク {i + 1}/{num_chunks} "
                f"({start / SAMPLE_RATE:.0f}s - {end / SAMPLE_RATE:.0f}s, "
                f"{chunk_dur:.0f}秒) ---"
            )

        # batch + channel 次元を追加 [B, T] → [B, 1, T]
        if chunk.ndim == 2:
            chunk_3d = chunk[:, None, :]
        else:
            chunk_3d = chunk

        # --- Acoustic エンコード ---
        acoustic_tokens = model.acoustic_tokenizer.encode(chunk_3d)
        mx.eval(acoustic_tokens)
        acoustic_feat = model.acoustic_connector(acoustic_tokens)
        mx.eval(acoustic_feat)

        del acoustic_tokens
        mx.metal.clear_cache()

        acoustic_features_list.append(acoustic_feat)

        if verbose:
            print(f"  Acoustic特徴: {acoustic_feat.shape}")
            print(
                f"  メモリ: {mx.metal.get_active_memory() / 1e9:.2f} GB "
                f"(ピーク: {mx.metal.get_peak_memory() / 1e9:.2f} GB)"
            )

        # --- Semantic エンコード ---
        semantic_tokens = model.semantic_tokenizer.encode(chunk_3d)
        mx.eval(semantic_tokens)
        semantic_feat = model.semantic_connector(semantic_tokens)
        mx.eval(semantic_feat)

        del semantic_tokens
        del chunk_3d
        mx.metal.clear_cache()

        semantic_features_list.append(semantic_feat)

        if verbose:
            print(f"  Semantic特徴: {semantic_feat.shape}")
            print(f"  メモリ: {mx.metal.get_active_memory() / 1e9:.2f} GB")

        gc.collect()

    # 全チャンクの特徴量を時間軸方向に結合
    print(f"\n特徴量結合中...")
    all_acoustic = mx.concatenate(acoustic_features_list, axis=1)
    mx.eval(all_acoustic)
    del acoustic_features_list

    all_semantic = mx.concatenate(semantic_features_list, axis=1)
    mx.eval(all_semantic)
    del semantic_features_list

    # Acoustic + Semantic を加算
    combined = all_acoustic + all_semantic
    mx.eval(combined)
    del all_acoustic, all_semantic
    mx.metal.clear_cache()
    gc.collect()

    print(f"結合済み特徴量: {combined.shape}")
    print(f"メモリ: {mx.metal.get_active_memory() / 1e9:.2f} GB")

    return combined


# ================================================================
# テキスト生成
# ================================================================

def generate_with_precomputed_features(
    model,
    speech_features: mx.array,
    audio_duration: float,
    context: str = None,
    max_tokens: int = MAX_TOKENS,
    prefill_step_size: int = PREFILL_STEP_SIZE,
    verbose: bool = True,
) -> dict:
    """
    事前計算済みの speech_features を使って、LLM でテキスト生成する。
    model.generate() の内部ロジックを再現するが、encode_speech をスキップ。
    """
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    print(f"\n=== テキスト生成 ===")
    print(f"音声長: {audio_duration:.1f}秒")
    print(f"特徴量トークン数: {speech_features.shape[1]}")

    # プロンプト構築
    input_ids, acoustic_input_mask = model._build_prompt_tokens(
        speech_features, audio_duration, context
    )
    print(f"入力トークン数: {input_ids.shape[1]}")

    # サンプラーとプロセッサ
    sampler = make_sampler(0.0, 1.0, 0.0, min_tokens_to_keep=1, top_k=0)
    logits_processors = make_logits_processors(
        repetition_penalty=1.0,
        repetition_context_size=100,
    )

    # トークン生成
    generated_tokens = []
    start_time = time.time()

    print("生成中...")
    for token, _ in model.stream_generate(
        input_ids=input_ids,
        speech_features=speech_features,
        acoustic_input_mask=acoustic_input_mask,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        prefill_step_size=prefill_step_size,
        verbose=verbose,
    ):
        generated_tokens.append(token)

    end_time = time.time()

    mx.clear_cache()

    # デコード
    text = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    segments = model.parse_transcription(text)

    total_time = end_time - start_time
    gen_tps = len(generated_tokens) / total_time if total_time > 0 else 0

    print(f"\n=== 完了 ===")
    print(f"生成トークン数: {len(generated_tokens)}")
    print(f"生成速度: {gen_tps:.1f} tokens/sec")
    print(f"処理時間: {total_time:.1f}秒")
    print(f"ピークメモリ: {mx.metal.get_peak_memory() / 1e9:.2f} GB")

    return {
        "text": text.strip(),
        "segments": segments,
        "generation_tokens": len(generated_tokens),
        "total_time": total_time,
        "generation_tps": gen_tps,
    }


# ================================================================
# メイン
# ================================================================

def main():
    print("=" * 60)
    print("VibeVoice-ASR 長時間音声書き起こし（メモリ最適化版）")
    print("=" * 60)

    # GPU最大推奨サイズの90%をメモリ上限に設定（OS分を確保）
    device_info = mx.metal.device_info()
    max_recommended = device_info["max_recommended_working_set_size"]
    memory_limit = int(max_recommended * 0.90)
    mx.metal.set_memory_limit(memory_limit)
    print(f"\nGPU最大推奨メモリ: {max_recommended / (1024**3):.1f} GB")
    print(f"メモリ上限設定: {memory_limit / (1024**3):.1f} GB")

    mx.reset_peak_memory()

    # モデルロード
    print("\nモデルロード中...")
    model = load_model(MODEL_ID)
    mx.eval(model.parameters())
    model_memory_gb = mx.metal.get_active_memory() / (1024 ** 3)
    print(f"モデルロード後メモリ: {model_memory_gb:.2f} GB")

    # 空きメモリを計測してチャンクサイズを自動決定
    available_gb = get_available_memory_gb()
    chunk_seconds = calculate_encoder_chunk_seconds(available_gb)
    print(f"\n=== チャンクサイズ自動決定 ===")
    print(f"空きメモリ: {available_gb:.2f} GB")
    print(f"推奨チャンクサイズ: {chunk_seconds}秒 ({chunk_seconds / 60:.1f}分)")

    # 音声前処理
    audio_tensor = load_and_preprocess_audio(model, AUDIO_PATH)
    audio_duration = audio_tensor.shape[1] / SAMPLE_RATE

    # チャンクエンコード（メモリスパイク回避の核心部分）
    speech_features = chunked_encode_speech(
        model,
        audio_tensor,
        chunk_seconds=chunk_seconds,
        verbose=True,
    )

    # 音声テンソルはもう不要
    del audio_tensor
    mx.metal.clear_cache()
    gc.collect()

    # テキスト生成（LLM部分は元々メモリ効率が良い）
    result = generate_with_precomputed_features(
        model,
        speech_features,
        audio_duration,
        prefill_step_size=PREFILL_STEP_SIZE,
        verbose=True,
    )

    # テキスト保存
    txt_path = f"{OUTPUT_PATH}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"\nテキスト保存: {txt_path}")

    # JSON保存（セグメント・統計情報付き）
    json_path = f"{OUTPUT_PATH}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "text": result["text"],
                "segments": result["segments"],
                "stats": {
                    "generation_tokens": result["generation_tokens"],
                    "total_time": result["total_time"],
                    "generation_tps": result["generation_tps"],
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"JSON保存: {json_path}")

    # 結果プレビュー
    print(f"\n=== 書き起こし結果（先頭500文字） ===")
    print(result["text"][:500])
    if result["segments"]:
        print(f"\n=== セグメント数: {len(result['segments'])} ===")
        for seg in result["segments"][:3]:
            print(seg)
        if len(result["segments"]) > 3:
            print("...")


if __name__ == "__main__":
    main()
