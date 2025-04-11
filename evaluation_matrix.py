import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Synthetic movie dataset with emotion tags
movies = {
    "Inception": "excited",
    "The Notebook": "romantic",
    "Titanic": "sad",
    "Interstellar": "awe",
    "The Dark Knight": "thrill",
    "Inside Out": "happy",
    "Joker": "dark",
    "Up": "nostalgic",
    "Toy Story": "joy",
    "Schindler's List": "emotional"
}

# Unique emotions
unique_emotions = list(set(movies.values()))

# Generate synthetic user input and actual labels for evaluation
true_labels = np.random.choice(unique_emotions, 20)  # Actual emotions user expected
predicted_labels = np.random.choice(unique_emotions, 20)  # Emotions from recommendations

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_emotions)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=unique_emotions, yticklabels=unique_emotions, cmap="Blues")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.title("Confusion Matrix for Emotion-Based Recommendations")
plt.show()

# Classification report
report = classification_report(true_labels, predicted_labels, target_names=unique_emotions)
print("Classification Report:\n", report)
