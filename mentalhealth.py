import pandas as pd
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')

# Load dataset (ensure correct path)
df = pd.read_csv(r"C:\Users\HP\Desktop\codealpha\chatbot\Mental_Health_FAQ.csv")

# Handling missing values
df = df.dropna()

# Convert questions and answers into lists
questions = df["Questions"].tolist()
answers = df["Answers"].tolist()

# TF-IDF Vectorizer for NLP-based matching
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)  # Train on predefined questions

# Function to get the best chatbot response
def chatbot_response(user_input):
    user_input = user_input.lower()
    input_vec = vectorizer.transform([user_input])  # Transform user input

    # Compute cosine similarity
    similarities = cosine_similarity(input_vec, X)
    best_match = similarities.argmax()  # Get the best match index

    if similarities[0, best_match] > 0.3:  # Set a threshold
        return random.choice(answers[best_match].split("|"))  # Return a random response
    else:
        return "I'm here to help. Could you elaborate on your feelings?"

# Chat loop
print("Mental Health Chatbot: Hi! I'm here to listen. Type 'exit' to end.")
while True:
    user_message = input("You: ")
    if user_message.lower() in ["exit", "quit", "bye"]:
        print("Bot: Take care! You're not alone.")
        break
    print("Bot:", chatbot_response(user_message))
