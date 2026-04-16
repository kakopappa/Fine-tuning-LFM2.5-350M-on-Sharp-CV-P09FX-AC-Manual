# Fine-tuning LFM2.5-350M on Sharp CV-P09FX AC Manual
## Complete Setup & Run Guide (Windows)

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 or 3.11 | 3.12+ not fully supported by Unsloth yet |
| CUDA GPU | 6GB+ VRAM recommended (RTX 3060 / 4060 or better) |
| CUDA Toolkit | 11.8 or 12.1 — must match your `requirements.txt` line |
| Git | For cloning if needed |

Check your CUDA version:
```
nvcc --version
```
Or check in Python:
```python
import torch
print(torch.version.cuda)
```

---

## 2. Folder structure

Your `E:\llm-finetune\ac-finetune-dataset` folder should look like this:

```
ac-finetune-dataset\
├── ac_manual_chatml.jsonl     ← training dataset (one entry per line)
├── ac_manual_chatml.json      ← same dataset as JSON array
├── train.py                   ← fine-tuning script
├── inference.py               ← test your trained model
├── requirements.txt           ← dependencies
└── README.md                  ← this file
```

---

## 3. Create a virtual environment

Open a terminal in `E:\llm-finetune\ac-finetune-dataset` and run:

```bat
python -m venv venv
venv\Scripts\activate
```

Your prompt should now show `(venv)`.

---

## 4. Install dependencies

### Step 4a — Install Unsloth

Pick the line that matches your CUDA version and uncomment it in `requirements.txt`, then run:

```bat
pip install "unsloth[cu121-torch260]"
```

Replace `cu121` with `cu118` if you have CUDA 11.8.

> **No GPU / CPU only?**
> ```bat
> pip install "unsloth[cpu]"
> ```
> Training will work but will be very slow (~hours vs ~minutes on GPU).

### Step 4b — Install remaining packages

```bat
pip install trl>=0.13.0 datasets>=2.18.0 huggingface_hub>=0.23.0 hf_transfer bitsandbytes>=0.43.0
```

### Step 4c — Enable fast HuggingFace downloads (optional but recommended)

```bat
set HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## 5. Verify GPU is detected

```bat
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output (example):
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4060
```

---

## 6. Run training

```bat
python train.py
```

What happens:
1. Downloads `LiquidAI/LFM2.5-350M-Instruct` from HuggingFace (~700MB, cached after first run)
2. Applies QLoRA adapters (PEFT)
3. Trains for 3 epochs on 45 examples
4. Saves LoRA adapters to `./lfm25_350m_ac_lora/`

Estimated training time:
| Hardware | Time |
|---|---|
| RTX 3060 / 4060 (8GB) | ~3–6 minutes |
| RTX 3080 / 4070 (12GB) | ~2–4 minutes |
| CPU only | ~60–120 minutes |

---

## 7. Test your fine-tuned model

```bat
python inference.py
```

This runs a few built-in test questions then enters interactive mode where you can ask anything about the AC manual.

---

## 8. Customizing training

Edit these values at the top of `train.py`:

| Parameter | Default | Description |
|---|---|---|
| `NUM_EPOCHS` | `3` | More epochs = better fit but risk of overfitting on small dataset |
| `LORA_RANK` | `16` | Higher = more trainable params, more VRAM. Try 8 for low VRAM |
| `LEARNING_RATE` | `2e-4` | Standard for LoRA. Lower (1e-4) if loss is unstable |
| `LOAD_IN_4BIT` | `True` | Set `False` if you have 16GB+ VRAM for slightly better quality |
| `MAX_STEPS` | `-1` | Set e.g. `60` to do a quick test run |

---

## 9. Save the full merged model (optional)

If you want a standalone model without needing the base model separately, uncomment this block in `train.py`:

```python
model.save_pretrained_merged(MERGED_SAVE_DIR, tokenizer, save_method="merged_16bit")
```

---

## 10. Export to GGUF for llama.cpp (optional)

To run the model offline with llama.cpp, uncomment this block in `train.py`:

```python
model.save_pretrained_gguf("lfm25_ac_gguf", tokenizer, quantization_method="q4_k_m")
```

Then run with llama.cpp:
```bat
llama-cli --model lfm25_ac_gguf\lfm25_ac_gguf-Q4_K_M.gguf --jinja --temp 0.1 --top-k 50 --top-p 0.1
```

---

## 11. Troubleshooting

**`CUDA out of memory`**
- Reduce `PER_DEVICE_TRAIN_BATCH_SIZE` to `1`
- Reduce `LORA_RANK` to `8`
- Make sure `LOAD_IN_4BIT = True`

**`ModuleNotFoundError: unsloth`**
- Make sure your venv is activated: `venv\Scripts\activate`
- Re-run the pip install step

**`bitsandbytes` errors on Windows**
- Install the Windows-compatible build: `pip install bitsandbytes --prefer-binary`

**Model download is slow**
- Set `set HF_HUB_ENABLE_HF_TRANSFER=1` before running

**`trust_remote_code` warning**
- LFM2.5 uses a custom architecture. Unsloth handles this automatically — it is safe to proceed.

---

## Dataset format reference

Each line in `ac_manual_chatml.jsonl` is:
```json
{
  "text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"
}
```

To add more training examples, append additional lines to `ac_manual_chatml.jsonl` in the same format.
