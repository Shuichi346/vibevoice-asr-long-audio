# VibeVoice-ASR 長時間音声メモリ最適化スクリプト

Mac Mini (Apple Silicon) のユニファイドメモリ 24〜25GB 環境で、**長時間音声**を VibeVoice-ASR 4bit/8bit モデルで書き起こすためのスクリプトです。

---

## 何が問題だったか

[VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) は Microsoft が開発した音声認識モデルで、**最大60分の音声を一括処理**できることが最大の特長です。MLX 向け量子化モデル（[mlx-community/VibeVoice-ASR-4bit](https://huggingface.co/mlx-community/VibeVoice-ASR-4bit)）を使えば、Apple Silicon Mac でもローカル実行できます。

しかし、**25GB ユニファイドメモリの Mac Mini で55分の音声を処理するとメモリエラー（OOM）で落ちる**という問題がありました。

### 症状の詳細

- 55分の音声をそのまま投入 → **メモリエラーでクラッシュ**
- 25分に分割すると動くが、**メモリ使用量がギリギリ**
- 不思議なことに、**処理中（LLM生成中）はメモリが低く安定**している
- メモリが爆発するのは**読み込みと処理開始時の一瞬だけ**

4bit モデルなら本来24GBで動くはず。README にもそう書いてある。なのに落ちる。

### 原因の特定

ソースコードを調査した結果、**メモリスパイクの原因は Audio Encoder（音声エンコーダ）** であることが判明しました。

VibeVoice-ASR の内部処理は以下の流れです：

```
[音声 55分 = 79,200,000サンプル]
        │
        ▼
[Audio Encoder]  ← ★ ここでメモリ爆発
  Acoustic Tokenizer (Conv1d 多段ダウンサンプリング)
  Semantic Tokenizer (同上)
        │
        ▼
[LLM (Qwen2)]  ← ここはメモリ効率が良い（KV-cache のみ）
        │
        ▼
[書き起こしJSON]
```

Audio Encoder は Conv1d の多段畳み込みで構成されており、各段で中間テンソルが生成されます。55分の音声（7920万サンプル）を一括投入すると、**中間バッファだけで25〜30GBに達し**、モデル重量（4bit ≈ 5GB）と合わせてメモリを超過します。

処理後の LLM 生成時は KV-cache しか使わないので、メモリは低く安定します。これが「処理中は大丈夫なのに開始時だけ爆発する」という症状の正体です。

---

## 解決策：エンコーダだけチャンク化し、LLM には全体を一括で渡す

このスクリプトの核心は**「音声を分割して個別に書き起こす」のではなく、「エンコーダ段階でのみチャンク分割し、LLM には結合した特徴量を一括で渡す」**ことです。

```
[音声 55分]
     │
     ▼  ← ★ ここだけチャンク化（10分ずつ）
[Audio Encoder] → 特徴量 chunk1, chunk2, chunk3, ...
     │
     ▼  ← 時間軸方向に結合
[55分ぶんの統合特徴量テンソル]
     │
     ▼  ← ★ ここは一括（分割なし）
[Qwen2 LLM が55分全体を一度に見て JSON 生成]
     │
     ▼
[{"Speaker":0, "Start":0, ...}, {"Speaker":1, ...}, ...]
```

### チャンクサイズの自動決定

モデルロード後の空きメモリを計測し、実測データに基づいた線形比例で安全なチャンクサイズを自動決定します。

24GB 環境での実測値（モデルロード後の空き約14GB → 600秒が適正）を基準とし、80% の安全マージンを適用します。

```
推奨秒数 = 600 × (実際の空きGB / 14GB) × 0.80
```

| 環境 | モデル後の空き | 推奨チャンクサイズ |
|---|---|---|
| 24GB Mac Mini | ~14 GB | 480秒（8分） |
| 36GB Mac | ~26 GB | 891秒（約15分） |
| 64GB Mac | ~54 GB | 1800秒（上限30分） |
| 16GB Mac (4bit) | ~4 GB | 137秒（約2分） |

---

## スピーカーIDは入れ替わらないか？

**入れ替わりません。** 最初から最後まで一貫します。

スピーカーIDを決定しているのは **LLM（Qwen2）** です。Audio Encoder は「音の特徴を数値ベクトルに変換する」だけで、誰が誰かという判断は一切しません。畳み込み演算なので、10分ごとに切ってエンコードしても、結合すれば一括エンコードと同じ数値列になります。

LLM は結合済みの55分全体の特徴量を受け取って、最初から最後まで通しで「この声は Speaker 0、この声は Speaker 1」と追跡します。

もし「25分ずつ分割して2回 `model.generate()` を呼ぶ」方式だったら、IDがリセットされて入れ替わるリスクがあります。このスクリプトはそうしていません。

---

## MAX_TOKENS について

LLM の生成トークン数上限は `MAX_TOKENS = 32768` で設定しています。

この値は `mlx_lm.generate.generate_step()` にそのまま渡され、ループ内で `n == max_tokens` に達すると停止します。また、VibeVoice-ASR は EOS トークン（`<|endoftext|>` / `<|im_end|>`）検出でも停止するため、通常は上限に達する前に自然終了します。

55分の日本語音声で概ね 4,000〜12,000 トークン程度（JSON フォーマットのタイムスタンプやスピーカーID含む）なので、32768 で十分です。

---

## 動作環境

- macOS（Apple Silicon M1/M2/M3/M4）
- Python 3.10+
- ユニファイドメモリ 16GB 以上（24GB 以上推奨）

## セットアップ

```bash
pip install -U mlx-audio mlx-lm
```

## 使い方

スクリプト先頭の定数を環境に合わせて編集してください。

```python
MODEL_ID = "mlx-community/VibeVoice-ASR-4bit"   # または 8bit
AUDIO_PATH = "/path/to/your/audio.wav"
OUTPUT_PATH = "/path/to/output"
```

実行：

```bash
python vibevoice_long_audio.py
```

出力ファイル：

- `{OUTPUT_PATH}.txt` — プレーンテキスト
- `{OUTPUT_PATH}.json` — セグメント情報（スピーカーID・タイムスタンプ）付き

## 調整可能なパラメータ

| 定数 | 説明 | デフォルト |
|---|---|---|
| `CHUNK_SAFETY_MARGIN` | 安全マージン。1.0 で実測ギリギリ、0.8 で余裕あり | `0.80` |
| `PREFILL_STEP_SIZE` | LLM プリフィルのステップサイズ。小さいほどメモリ節約 | `512` |
| `MAX_TOKENS` | LLM 生成トークン数の上限 | `32768` |
| `CHUNK_MIN_SECONDS` | チャンクの最小サイズ | `120`（2分） |
| `CHUNK_MAX_SECONDS` | チャンクの最大サイズ | `1800`（30分） |

## ライセンス

VibeVoice-ASR は [MIT License](https://huggingface.co/microsoft/VibeVoice-ASR) で公開されています。このスクリプトも MIT License とします。

## 参考リンク

- [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) — オリジナルモデル
- [mlx-community/VibeVoice-ASR-4bit](https://huggingface.co/mlx-community/VibeVoice-ASR-4bit) — MLX 4bit 量子化版
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX Audio ライブラリ
- [VibeVoice-ASR Technical Report](https://arxiv.org/pdf/2601.18184) — 技術レポート
