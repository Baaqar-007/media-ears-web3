
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-emotion dataset (synthetic emotions instead of ratings)
emotions = pd.DataFrame({
    'Inception': ['Excited', 'Intrigued', None, 'Bored', 'Neutral'],
    'Titanic': ['Sad', None, 'Heartfelt', 'Tense', None],
    'The Dark Knight': [None, 'Tense', 'Thrilled', 'Excited', 'Curious'],
    'Forrest Gump': ['Happy', None, 'Inspired', 'Emotional', 'Touched'],
    'The Matrix': [None, 'Amazed', 'Curious', None, 'Confused']
}, index=['Alice', 'Bob', 'Charlie', 'David', 'Emma'])

print("User-Emotion Matrix (None means no recorded emotion):")
print(emotions)

# Encode emotions numerically for similarity computation
emotion_mapping = {
    'Excited': 5, 'Intrigued': 4, 'Bored': 1, 'Neutral': 2, 'Sad': 1,
    'Heartfelt': 4, 'Tense': 3, 'Thrilled': 5, 'Curious': 4, 'Happy': 5,
    'Inspired': 4, 'Emotional': 3, 'Touched': 4, 'Amazed': 5, 'Confused': 2
}

def encode_emotions(df):
    return df.applymap(lambda x: emotion_mapping.get(x, 0))

encoded_emotions = encode_emotions(emotions)

# Compute similarity between users
user_similarity = cosine_similarity(encoded_emotions.fillna(0))
np.fill_diagonal(user_similarity, 0)  # Ignore self-similarity

print("\nUser Similarity Matrix:")
print(pd.DataFrame(user_similarity, index=emotions.index, columns=emotions.index))

# Function to get recommendations based on entered emotion
def get_recommendations(user_name, emotion, top_n=2):
    if user_name not in emotions.index:
        print("Invalid user. Please enter a valid name.")
        return []
    
    user_index = emotions.index.get_loc(user_name)
    similar_users = np.argsort(user_similarity[user_index])[::-1]
    
    # Find movies with the desired emotion
    relevant_movies = []
    for movie in emotions.columns:
        if emotions.loc[user_name, movie] is None:
            for similar_user in similar_users:
                if emotions.iloc[similar_user][movie] == emotion:
                    relevant_movies.append(movie)
                    break
    
    return relevant_movies[:top_n]

# Interactive user input
user_name = input("Enter your name (Alice, Bob, Charlie, David, Emma): ")
emotion = input("Enter an emotion (Excited, Happy, Sad, etc.): ")
recommendations = get_recommendations(user_name, emotion)
if recommendations:
    print(f"\nRecommended movies for {user_name} based on emotion '{emotion}': {recommendations}")
else:
    print(f"\nNo movies found for {user_name} based on emotion '{emotion}'")
