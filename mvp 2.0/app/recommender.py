import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load model and data
model = joblib.load("model_emotion.pkl")
df = pd.read_csv("podcast_data_real.csv")

# Create text column and drop rows with missing values
df["text"] = (df["title"] + " " + df["description"]).fillna("")
df = df.dropna(subset=["text"])

# TF-IDF vectorizer setup
vectorizer = TfidfVectorizer(stop_words="english")

def recommend(text_input, top_n=5):
    emotion = model.predict([text_input])[0]
    print(emotion)
    
    filtered_df = df[df["emotion_tag"] == emotion].copy()
    
    if filtered_df.empty:
        return []

    # Ensure clean input for TF-IDF
    filtered_df["text"] = filtered_df["text"].fillna("")
    combined_text = list(filtered_df["text"]) + [text_input]
    
    tfidf_input = vectorizer.fit_transform(combined_text)
    sim_scores = cosine_similarity(tfidf_input[-1], tfidf_input[:-1]).flatten()
    
    top_indices = sim_scores.argsort()[::-1][:top_n]
    return filtered_df.iloc[top_indices][["title", "description"]].to_dict(orient="records")
