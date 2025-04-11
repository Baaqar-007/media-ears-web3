# app.py
from flask import Flask, request, jsonify, render_template
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import csv
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# ======= Model 1: Emotion Extraction ======= #
model_path = "emotion-model"  # Path to your saved emotion model
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
emotion_model = DistilBertForSequenceClassification.from_pretrained(model_path)
emotion_model.eval()

# Emotion labels should match your training labels.
LABEL_MAP = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        logits = outputs.logits
        pred_id = logits.argmax().item()
    return LABEL_MAP[pred_id]

# ======= Load Song Database ======= #
def load_song_db(file="songs.csv"):
    songs = []
    with open(file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            songs.append(row)
    return songs

SONG_DB = load_song_db()

# ======= Pre-compute Lyric Embeddings ======= #
# Initialize a SentenceTransformer model for lyric matching.
lyric_model = SentenceTransformer('all-MiniLM-L6-v2')
# Compute and store embeddings for each song's lyrics.
for song in SONG_DB:
    # Ensure the lyrics field exists
    text_lyrics = song.get("lyrics", "")
    song["lyrics_embedding"] = lyric_model.encode(text_lyrics, convert_to_tensor=True)

# ======= Recommendation Functions ======= #
def recommend_songs(emotion, limit=5):
    """Filter songs from SONG_DB based on emotion."""
    recommended = [song for song in SONG_DB if song["emotion"].strip().lower() == emotion]
    return recommended[:limit]

def recommend_songs_by_lyrics(user_text, limit=5):
    """Recommend songs based on similarity between user text and song lyrics."""
    input_embedding = lyric_model.encode(user_text, convert_to_tensor=True)
    similarities = []
    for song in SONG_DB:
        sim = util.cos_sim(input_embedding, song["lyrics_embedding"]).item()
        similarities.append((sim, song))
    # Sort songs by descending similarity score.
    similarities.sort(key=lambda x: x[0], reverse=True)
    # Return top 'limit' songs
    top_songs = [song for _, song in similarities][:limit]
    return top_songs

# ======= History & Similar Songs ======= #
HISTORY_FILE = 'history.json'
try:
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        history_data = json.load(f)
except FileNotFoundError:
    history_data = []

def log_play(song, emotion):
    entry = {
        'title': song['title'],
        'artist': song['artist'],
        'emotion': emotion,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    history_data.append(entry)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=2)

def get_similar_songs():
    """Use the emotion of the most recent play to fetch similar songs (emotion filtering)."""
    if history_data:
        recent_emotion = history_data[-1]["emotion"]
        similar = recommend_songs(recent_emotion, limit=5)
        return similar
    return []

# ======= Flask Routes ======= #
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_text = request.form["text"]
        # Model 1: Emotion prediction
        emotion = predict_emotion(user_text)
        # Model 2a: Emotion-based recommendation
        emotion_based_songs = recommend_songs(emotion)
        # Model 2b: Lyric matching recommendation (based on user text)
        lyrics_based_songs = recommend_songs_by_lyrics(user_text)
        # Log the first recommended song from emotion filter (for history demo)
        if emotion_based_songs:
            log_play(emotion_based_songs[0], emotion)
        similar_songs = get_similar_songs()
        return render_template("index.html", 
                               text=user_text,
                               emotion=emotion, 
                               emotion_songs=emotion_based_songs,
                               lyrics_songs=lyrics_based_songs,
                               similar_songs=similar_songs)
    else:
        similar_songs = get_similar_songs()
        return render_template("index.html", similar_songs=similar_songs)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    user_text = data.get("text", "")
    emotion = predict_emotion(user_text)
    emotion_songs = recommend_songs(emotion)
    lyrics_songs = recommend_songs_by_lyrics(user_text)
    return jsonify({
        "emotion": emotion,
        "emotion_songs": emotion_songs,
        "lyrics_songs": lyrics_songs,
    })

if __name__ == "__main__":
    app.run(debug=True)
