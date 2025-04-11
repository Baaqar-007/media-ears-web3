# train_emotion_model.py
# MVP: simple text-to-emotion classification using TF-IDF + Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv("..\\app\\podcast_data_real.csv")

# Use description for training

df["text"] = df["title"] + " " + df["description"]
df["text"] = df["text"].fillna(df["description"].fillna("No content available"))



df = df.dropna(subset=["emotion_tag"])
print("Kept rows:", len(df))
print("Dropped rows:", df["emotion_tag"].isna().sum())

X = df["text"]
y = df["emotion_tag"]

# Split
txt_train, txt_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=500, random_state=42, n_jobs=-1))
])
print(df['emotion_tag'].value_counts())

# Train
pipeline.fit(txt_train, y_train)

# Evaluate
acc = pipeline.score(txt_test, y_test)
print(f"Validation accuracy: {acc:.2f}")

# Save model
import os
joblib.dump(pipeline, os.path.join(os.getcwd(), "model_emotion.pkl"))
