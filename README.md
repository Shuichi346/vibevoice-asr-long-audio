# VibeVoice-ASR Long Audio Memory Optimization Script

A script for transcribing **long audio files of 55+ minutes** using VibeVoice-ASR 4bit/8bit models in a Mac Mini (Apple Silicon) unified memory environment of 24-25GB.

---

## What Was the Problem

[VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) is a speech recognition model developed by Microsoft, with its key feature being **batch processing of up to 60 minutes of audio**. Using the quantized model for MLX ([mlx-community/VibeVoice-ASR-4bit](https://huggingface.co/mlx-community/VibeVoice-ASR-4bit)), it can run locally on Apple Silicon Macs.

However, there was an issue where **processing 55 minutes of audio on a Mac Mini with 25GB unified memory would crash with out-of-memory (OOM) errors**.

### Detailed Symptoms

- Feeding 55 minutes of audio directly → **Crashes with memory error**
- Works when split into 25-minute segments, but **memory usage is at the limit**
- Strangely, **memory remains low and stable during processing (LLM generation)**
- Memory explosion occurs only **for a brief moment during loading and processing initiation**

The 4bit model should theoretically work with 24GB. The README states so. Yet it crashes.

### Root Cause Identification

After investigating the source code, it was discovered that **the cause of memory spikes is the Audio Encoder**.

The internal processing flow of VibeVoice-ASR is as follows:

```
[Audio 55min = 79,200,000 samples]
        │
        ▼
[Audio Encoder]  ← ★ Memory explosion occurs here
  Acoustic Tokenizer (Multi-stage Conv1d downsampling)
  Semantic Tokenizer (Same as above)
        │
        ▼
[LLM (Qwen2)]  ← Memory efficient here (KV-cache only)
        │
        ▼
[Transcription JSON]
```

The Audio Encoder consists of multi-stage Conv1d convolutions, generating intermediate tensors at each stage. When 55 minutes of audio (79.2 million samples) is fed in at once, **intermediate buffers alone reach 25-30GB**, exceeding memory when combined with model weights (4bit ≈ 5GB).

During subsequent LLM generation, only KV-cache is used, so memory remains low and stable. This is the true nature of the "fine during processing but explodes only at startup" symptom.

---

## Solution: Chunk Only the Encoder, Pass Complete Data to LLM

The core concept of this script is **not "splitting audio for separate transcription" but "chunk splitting only at the encoder stage, then passing combined features to LLM as a batch"**.

```
[Audio 55min]
     │
     ▼  ← ★ Chunking only here (10min each)
[Audio Encoder] → Features chunk1, chunk2, chunk3, ...
     │
     ▼  ← Concatenate along time axis
[Integrated feature tensor for 55min]
     │
     ▼  ← ★ Batch processing here (no splitting)
[Qwen2 LLM sees entire 55min at once and generates JSON]
     │
     ▼
[{"Speaker":0, "Start":0, ...}, {"Speaker":1, ...}, ...]
```

### Automatic Chunk Size Determination

After model loading, available memory is measured and safe chunk size is automatically determined based on linear proportion from empirical data.

Based on measured values in 24GB environment (approximately 14GB free after model loading → 600 seconds is appropriate), applying an 80% safety margin.

```
Recommended seconds = 600 × (Actual free GB / 14GB) × 0.80
```

| Environment | Free after model | Recommended chunk size |
|---|---|---|
| 24GB Mac Mini | ~14 GB | 480 seconds (8min) |
| 36GB Mac | ~26 GB | 891 seconds (~15min) |
| 64GB Mac | ~54 GB | 1800 seconds (30min max) |
| 16GB Mac (4bit) | ~4 GB | 137 seconds (~2min) |

---

## Do Speaker IDs Get Mixed Up?

**No, they don't.** They remain consistent from beginning to end.

Speaker ID determination is handled by the **LLM (Qwen2)**. The Audio Encoder only "converts audio features to numerical vectors" and makes no judgments about who is who. Since it's convolution operations, encoding in 10-minute chunks and then concatenating produces the same numerical sequence as batch encoding.

The LLM receives the concatenated 55-minute feature data and tracks "this voice is Speaker 0, this voice is Speaker 1" consistently from start to finish.

If using a "split into 25-minute segments and call `model.generate()` twice" approach, there would be risk of ID reset and mix-ups. This script doesn't do that.

---

## About MAX_TOKENS

The LLM generation token limit is set to `MAX_TOKENS = 32768`.

This value is passed directly to `mlx_lm.generate.generate_step()` and stops when `n == max_tokens` is reached in the loop. Additionally, VibeVoice-ASR also stops upon EOS token detection (`<|endoftext|>` / `<|im_end|>`), so it typically terminates naturally before reaching the limit.

For 55 minutes of Japanese audio, approximately 4,000-12,000 tokens (including JSON format timestamps and speaker IDs) are typical, so 32768 is sufficient.

---

## Operating Environment

- macOS (Apple Silicon M1/M2/M3/M4)
- Python 3.10+
- Unified memory 16GB or more (24GB+ recommended)

## Setup

```bash
pip install -U mlx-audio mlx-lm
```

## Usage

Edit the constants at the beginning of the script to match your environment.

```python
MODEL_ID = "mlx-community/VibeVoice-ASR-4bit"   # or 8bit
AUDIO_PATH = "/path/to/your/audio.wav"
OUTPUT_PATH = "/path/to/output"
```

Run:

```bash
python vibevoice_long_audio.py
```

Output files:

- `{OUTPUT_PATH}.txt` — Plain text
- `{OUTPUT_PATH}.json` — With segment information (speaker ID, timestamps)

## Adjustable Parameters

| Constant | Description | Default |
|---|---|---|
| `CHUNK_SAFETY_MARGIN` | Safety margin. 1.0 for measured limit, 0.8 for buffer | `0.80` |
| `PREFILL_STEP_SIZE` | LLM prefill step size. Smaller saves memory | `512` |
| `MAX_TOKENS` | LLM generation token limit | `32768` |
| `CHUNK_MIN_SECONDS` | Minimum chunk size | `120` (2min) |
| `CHUNK_MAX_SECONDS` | Maximum chunk size | `1800` (30min) |

## License

VibeVoice-ASR is published under [MIT License](https://huggingface.co/microsoft/VibeVoice-ASR). This script is also under MIT License.

## References

- [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) — Original model
- [mlx-community/VibeVoice-ASR-4bit](https://huggingface.co/mlx-community/VibeVoice-ASR-4bit) — MLX 4bit quantized version
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX Audio library
- [VibeVoice-ASR Technical Report](https://arxiv.org/pdf/2601.18184) — Technical report
