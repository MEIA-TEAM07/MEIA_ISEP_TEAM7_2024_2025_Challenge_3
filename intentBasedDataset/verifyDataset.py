import pandas as pd

# Load the dataset
df = pd.read_csv("agricultural_chatbot_dataset.csv")

# Show dataset structure
print(df.head())
print(df.info())  # Check if columns are correctly formatted