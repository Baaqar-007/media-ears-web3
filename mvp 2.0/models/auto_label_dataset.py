import pandas as pd
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import torch
import os

# Load and preprocess dataset
df = pd.read_csv("top_podcasts.csv").fillna("")

# Use a 100-row sample for speed
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# Construct meaningful input text
df["text"] = df["episodeName"] + " " + df["description"]
df["text"] = df.apply(
    lambda row: row["text"].strip() if row["text"].strip() else row["show.description"],
    axis=1
)

# Truncate to 512 characters (let tokenizer handle this)
texts = df["text"].tolist()

# Detect device (GPU if available)
device = 0 if torch.cuda.is_available() else -1

# Load emotion classifier pipeline
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1,
    truncation=True,
    device=device
)

tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
long_texts = [t for t in texts if len(tokenizer.encode(t)) > 512]
print("Texts over 512 tokens:", len(long_texts))


# Run inference with progress bar
print("🔍 Tagging ~1000 podcasts with emotion labels...")
results = []
for text in tqdm(texts, desc="Tagging"):
    res = classifier(text)
    results.append(res[0][0]['label'] if res else None)

# Attach results
df["emotion_tag"] = results
df = df.dropna(subset=["emotion_tag"])
df = df[["episodeName", "description", "emotion_tag"]]
df.columns = ["title", "description", "emotion_tag"]

# Save to CSV
output_path = os.path.join(os.getcwd(), "podcast_data_real.csv")
df.to_csv(output_path, index=False)
print(f"✅ Emotion-labeled dataset saved to: {output_path}")
