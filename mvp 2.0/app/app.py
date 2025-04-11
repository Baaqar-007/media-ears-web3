from flask import Flask, request, jsonify
import joblib
from recommender import recommend

app = Flask(__name__)
model = joblib.load("model_emotion.pkl")

from flask import send_from_directory

from flask import render_template

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/styles.css")
def styles():
    return send_from_directory("static", "styles.css")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    emotion = model.predict([text])[0]
    return jsonify({"emotion": emotion})

@app.route("/recommend", methods=["POST"])
def get_recommendations():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    results = recommend(text)
    return jsonify({"recommendations": results})

if __name__ == "__main__":
    app.run(debug=True)
