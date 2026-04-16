# Sharp CV-P09FX AC Manual Assistant

This notebook demonstrates fine-tuning the **LiquidAI/LFM2.5-350M** model using **Unsloth** and **QLoRA** to act as a specialized technical assistant for the Sharp CV-P09FX Portable Air Conditioner.

## 🚀 Project Overview
The goal is to transform a general-purpose lightweight language model into a domain-specific expert that provides accurate technical specifications (power requirements, fan modes, maintenance, etc.) based strictly on the product manual.

## 🛠️ Technical Configuration
To overcome the challenges of a small model (350M parameters) overriding its pre-trained knowledge, we utilized the following specialized settings:

*   **Framework**: [Unsloth](https://github.com/unslothai/unsloth) (2x faster fine-tuning).
*   **Quantization**: 4-bit QLoRA for efficient memory usage.
*   **LoRA Rank (r)**: 64 (Increased capacity for fact retention).
*   **LoRA Alpha**: 32.
*   **Epochs**: 30 (Extended training to ensure memorization of manual details).
*   **Learning Rate**: 2e-4.
*   **Prompt Format**: ChatML.

## 📂 Dataset Structure
The model is trained on a `.jsonl` file located at `./ac_manual_chatml.jsonl`. Each entry follows the ChatML format:
```json
{
  "text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"
}
```

## 📖 How to Use
1.  **Install Dependencies**: Run the first cell to install `unsloth`, `trl`, and `bitsandbytes`.
2.  **Fine-Tuning**: Run the training cell. It will load the base model, apply LoRA adapters, and save the result to `./lfm25_350m_ac_lora`.
3.  **Inference**: Run the inference cell to load the fine-tuned adapters. You can use the `ask()` function or the interactive loop to query the assistant.

## ⚠️ Important Note
If the model produces incorrect technical specs (e.g., wrong voltage), ensure the `SYSTEM_PROMPT` during inference matches the training prompt exactly: *\"Answer questions accurately based on the product manual.\"*
