# Sharp CV-P09FX AC Manual AI Assistant

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

<img width="1020" height="558" alt="image" src="https://github.com/user-attachments/assets/f64e49fc-486a-47fe-bb62-07e3718aeb71" />

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

## ⚠️ Verdict:

Can answer strightforward questions well.

**Question**:  What power supply does the Sharp CV-P09FX require?

**Answer**: The unit requires a three-pin grounded electrical AC socket rated 125V, 60Hz, and 15 amps or more. Do not use extension cords. The power plug has a built-in fuse and must always be checked before use using the RESET and TEST buttons on the plug.

**Question**: Whats the filter coverage for Canada warranty?

**Answer**: For the Canada warranty, the filter covers approximately 95% of the air quality in the unit. Filters must be replaced every two years. If you need to replace the filter before year end, you can cancel the warranty and receive full coverage for another two years.

**Question**:What is the warranty difference on the Sharp CV-P09FX in Canada VS USA?

**Answer**: The Canadian version has a broader warranty covering one year of parts and labor, but the USA version covers up to 3 years of parts and labor with a oficht certificate. The USA certificate is more detailed and requires the customer to complete a pre-registration process before purchasing.

**Question**: Can the unit freeze up and how do I prevent it?
 
**Answer**: Yes, freezing can occur when the unit is set close to 64°F in low ambient temperature conditions, especially at night. To prevent freezing, set the unit to a higher temperature. If the cooling coil is already frozen, run the unit in Fan mode at HIGH speed until all ice has melted.


## 🎓 Lessons Learned & Summary

During the fine-tuning of the **Sharp CV-P09FX Assistant**, we discovered several critical factors for successfully specializing a small (350M) model:

### 1. The Capacity Challenge (LoRA Rank)
*   **Finding**: Standard LoRA ranks (e.g., 8 or 16) were insufficient for the model to accurately recall specific technical details like voltage or fan modes.
*   **Solution**: Increasing `LORA_RANK` to **64** provided the 'memory capacity' needed to store exact manual specifications without forgetting them.

### 2. Overcoming Pre-trained Bias (Epochs)
*   **Finding**: At 3 epochs, the model still defaulted to general knowledge (e.g., assuming a 12V power supply common in electronics).
*   **Solution**: Extending training to **30 epochs** allowed the model to successfully 'overwrite' its general weights with the specific **125V** requirements from the Sharp manual.

### 3. Prompt Alignment
*   **Finding**: If the `SYSTEM_PROMPT` used during inference differs even slightly from the one used in training, the model may hallucinate or fail to use its fine-tuned knowledge.
*   **Solution**: Maintain a strict 1:1 match between the training and inference system prompts.

### 4. Efficiency with Unsloth
*   **Finding**: Fine-tuning even for 30 epochs on a T4 GPU was remarkably fast thanks to Unsloth's optimizations and 4-bit quantization, making this workflow highly accessible for edge-case or domain-specific technical assistants.
