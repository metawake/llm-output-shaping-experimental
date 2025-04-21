import os
import openai
import anthropic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from collections import Counter
from datetime import datetime

# 🔑 Set API Keys and Endpoint
openai.api_key = os.getenv("OPENROUTER_API_KEY") or "org-your-openrouter-key-here"
openai.api_base = "https://openrouter.ai/api/v1"

# ============ Configuration ============
MODELS = {
    "mistralai/mistral-7b-instruct": "Mistral-7B",
    "openai/gpt-3.5-turbo": "GPT-3.5-Turbo",
    "anthropic/claude-3-haiku": "Claude-3-Haiku"
}

INTENTS = {
    "brief": "Give a concise, 1-2 sentence explanation.",
    "brief+60-words": "Give a brief explanation in around 60 words.",
    "minimal": "Use the fewest words possible.",
    "minimal+30-tokens": "Respond using no more than 30 tokens.",
    "business": "Answer in a professional business tone.",
    "clarity": "Explain clearly for an average reader.",
    "compact-bullet": "Summarize using bullet points only.",
    "casual": "Use a casual, friendly tone.",
    "legalish": "Respond in legal-sounding formal language.",
    "headline": "Answer using a headline-style phrase.",
    "qa": "Format as a Q&A with a clear answer.",
    "telegraph": "Use a telegraphic writing style.",
    "social-post": "Write it like a tweet or short social post.",
    "none": ""
}

PROMPTS = [
    "What is anemia?",
    "How does a blockchain work?",
    "How to cook rice?",
    "Why is the sky blue?",
    "What doctor do you recommend if I have bloated stomach and pain?",
    "I can't log into my work dashboard. What should I try?"
]

# ============ Utility Functions ============
def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def calc_entropy(text):
    tokens = text.split()
    probs = [tokens.count(tok)/len(tokens) for tok in tokens]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def detect_formatting(text):
    features = []
    if "\n- " in text or text.strip().startswith("-"):
        features.append("bullet")
    if any(text.strip().startswith(f"{n}.") for n in range(1, 5)):
        features.append("numbered")
    if text.lower().startswith("q:") or "a:" in text.lower():
        features.append("qa")
    if len(text) < 50:
        features.append("headline")
    return features

# ============ Main Script ============
results = []

def call_model(model_id, system_prompt, user_prompt):
    if "gpt-" in model_id:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        return completion.choices[0].message.content

    elif "claude" in model_id:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
        )
        return message.content[0].text

    elif "mistral" in model_id:
        completion = openai.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}],
        )
        return completion.choices[0].message.content

    else:
        return "[ERROR] Unknown model"

for model_id, model_label in MODELS.items():
    for intent_code, intent_instruction in INTENTS.items():
        for prompt in PROMPTS:
            full_prompt = prompt
            if intent_code != "none":
                full_prompt += f"\nInstruction: {intent_instruction}"
            try:
                output = call_model(model_id, "Answer the user request.", full_prompt)
                token_count = count_tokens(output)
                entropy = calc_entropy(output)
                fmt = detect_formatting(output)
                print(f"\n🧠 [{model_label}] [{intent_code.upper()}] {prompt}\n✍️ Output: {output}\n📏 Tokens: {token_count} | 📉 Entropy: {entropy:.2f}")
                results.append({
                    "model": model_label,
                    "intent": intent_code,
                    "prompt": prompt,
                    "output": output,
                    "token_count": token_count,
                    "entropy": entropy,
                    "formatting": ", ".join(fmt),
                    "format_count": len(fmt),
                    "length_hit": "<60w" in intent_code or "<30t" in intent_code and token_count < 60
                })
            except Exception as e:
                print(f"[ERROR] {e}")

# ============ Export ============
df = pd.DataFrame(results)
df.to_csv("model_outputs.csv", index=False)

# ============ Charts ============
plt.figure(figsize=(12,6))
for model in df["model"].unique():
    subset = df[df["model"] == model]
    avg_lengths = subset.groupby("intent")["token_count"].mean()
    plt.plot(avg_lengths.index, avg_lengths.values, marker='o', label=model)
plt.title("📏 Avg Output Length by Intent")
plt.xlabel("Intent")
plt.ylabel("Token Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output_lengths.png")
plt.close()

plt.figure(figsize=(12,6))
for model in df["model"].unique():
    subset = df[df["model"] == model]
    avg_entropy = subset.groupby("intent")["entropy"].mean()
    plt.plot(avg_entropy.index, avg_entropy.values, marker='s', label=model)
plt.title("📉 Avg Entropy by Intent")
plt.xlabel("Intent")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("entropy_by_intent.png")
plt.close()

format_counts = []
for (model, intent), group in df.groupby(["model", "intent"]):
    formats = Counter(", ".join(group["formatting"].dropna().values).split(", "))
    format_counts.append({"model": model, "intent": intent, **formats})
fmt_df = pd.DataFrame(format_counts).fillna(0)

fmt_df.set_index(["model", "intent"]).plot(kind="bar", stacked=True, figsize=(14,6))
plt.title("🧱 Formatting Features by Intent and Model")
plt.ylabel("Feature Count")
plt.tight_layout()
plt.savefig("formatting_features.png")

# Bonus Chart: Success of Length Constraint
length_control_df = df[df["intent"].str.contains("60|30")]
length_control_df["hit"] = length_control_df["token_count"] < 65
length_summary = length_control_df.groupby(["model", "intent"])["hit"].mean().unstack().fillna(0)
length_summary.plot(kind="bar", figsize=(10,6))
plt.title("🎯 Length Constraint Fulfillment by Model")
plt.ylabel("% Prompts Under Limit")
plt.tight_layout()
plt.savefig("length_success.png")
