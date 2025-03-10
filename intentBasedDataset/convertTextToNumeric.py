import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib  

# Load the dataset
df = pd.read_csv("agricultural_chatbot_dataset.csv")

# Show dataset structure
print(df.head())
print(df.info())  # Check if columns are correctly formatted

# Convert questions into TF-IDF numerical representation
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 most important words
X = vectorizer.fit_transform(df["question"])  # Transform questions into numerical form
y = df["intent"]  # The target labels (intent classification)

# Show the shape of the feature matrix
print("Feature matrix shape:", X.shape)  # Should be (95922, 5000)

# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

joblib.dump(X_train, "X_train.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")  # Save vectorizer for inference

print("Processed data saved! You can now run `trainModels.py` to train models.")