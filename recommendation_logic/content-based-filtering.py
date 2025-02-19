import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Expanded dataset - items and their descriptions
items = {
    "item1": "This is the first item",
    "item2": "Second item.",
    "item3": "This is the third item",
    "item4": "Out of Stock.",
    "item5": "Restricted.",
    "item6": "Sixth item.",
    "item7": "This item is currently unavailable.",
    "item8": "This is another unique item with special properties.",
    "item9": "Banned due to violation of policies.",
    "item10": "The first item is often the most important."
}

# Convert item descriptions into a TF-IDF matrix
vectorizer = TfidfVectorizer()
item_descriptions = list(items.values())
tfidf_matrix = vectorizer.fit_transform(item_descriptions)

# Display the TF-IDF matrix for understanding
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
print("Feature Names:", vectorizer.get_feature_names_out())

# Function to get recommendations for a given item
def get_recommendations(item_name, top_n=3):
    item_index = list(items.keys()).index(item_name)
    item_vector = tfidf_matrix[item_index]

    # Compute cosine similarity with all items
    similarities = cosine_similarity(item_vector, tfidf_matrix).flatten()

    # Display similarity scores for debugging
    similarity_dict = {list(items.keys())[i]: similarities[i] for i in range(len(items))}
    print(f"\nSimilarity Scores for '{item_name}':")
    for key, value in sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{key}: {value:.4f}")

    # Get top N similar items (excluding itself)
    similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
    similar_items = [list(items.keys())[i] for i in similar_indices]

    return similar_items

# Test the recommendation system
for i in range(len(items)):
    item_to_recommend_for = f"item{i+1}"
    recommendations = get_recommendations(item_to_recommend_for)
    print(f"\nRecommended items for '{item_to_recommend_for}': {recommendations}")
