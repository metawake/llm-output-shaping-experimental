# 🔍 LLM Output Shaping: Prompt Modifier Impact Explorer

This experimental project investigates how **prompt modifiers** influence the **length, structure, and entropy** of responses generated by large language models (LLMs). Our goal is to understand whether stylistic or instructional cues can **shrink or reformat model outputs** reliably — a practical interest for optimizing token costs and shaping user-facing text.

Full read is here: https://medium.com/@TheWake/prompt-shaping-measuring-the-impact-of-prompt-modifiers-on-output-size-and-format-9a53e06dccd6

## ✨ Key Features

- Compare **multiple models**: Mistral-7B, GPT-3.5-Turbo, Claude-3-Haiku (via [OpenRouter](https://openrouter.ai))
- Rich variety of **prompt modifiers**:
  - `"brief"`, `"minimal"`, `"business"`, `"casual"`, `"qa"`, `"telegraph"`, `"headline"`, `"social-post"`, etc.
- Analyze output for:
  - 📏 Token count  
  - 📉 Entropy  
  - 🧱 Formatting features (bullets, newlines, QA style)
- Export to:
  - `model_outputs.csv` (raw results)
  - `output_lengths.png`, `entropy_by_intent.png`, `formatting_features.png`, `length_success.png`

## 📊 Visual Output

The script produces multiple charts for clear comparison:

| Chart | Description |
|-------|-------------|
| `output_lengths.png` | Average output length across prompts and modifiers |
| `entropy_by_intent.png` | Token entropy to approximate verbosity or repetitiveness |
| `formatting_features.png` | Count of structural features in LLM responses |
| `length_success.png` | How often length constraints were fulfilled |

## 🚀 How to Run

1. Install dependencies:

   ```bash
   pip install openai anthropic tiktoken pandas matplotlib
