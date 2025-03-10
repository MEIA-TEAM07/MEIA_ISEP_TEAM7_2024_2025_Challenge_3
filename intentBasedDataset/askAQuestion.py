import joblib

# Load the trained model and vectorizer
svm_model = joblib.load("svm_model.pkl")  # Load SVM model
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Load the vectorizer

# Function to classify new questions
def classify_question(question):
    X_new = vectorizer.transform([question])  # Convert text to numerical features
    prediction = svm_model.predict(X_new)[0]  # Predict intent
    return prediction

# Test with user input
print("\n🔹 Ask a question about fruits (e.g., 'Is this banana good to eat?')")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Goodbye! 👋")
        break
    predicted_intent = classify_question(user_input)
    print(f"🔹 Predicted Intent: {predicted_intent}")