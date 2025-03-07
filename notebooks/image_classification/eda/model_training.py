import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import cv2
import os
from pathlib import Path

class DataLoader:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.datasets = ['train', 'valid', 'test']
        
    def load_dataset(self, dataset_type):
        if dataset_type not in self.datasets:
            raise ValueError(f"Dataset type must be one of {self.datasets}")
        
        # Load CSV file
        csv_path = self.base_path / dataset_type / '_classes.csv'
        df = pd.read_csv(csv_path)
        
        # Extract features and labels
        X = []
        valid_indices = []
        
        # Load and preprocess images
        for idx, filename in enumerate(df['filename'].values):
            img_path = self.base_path / dataset_type / filename
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, (224, 224))
                img_array = img.flatten()
                X.append(img_array)
                valid_indices.append(idx)
            else:
                print(f"Warning: Could not load image {filename}")
        
        # Convert to numpy arrays and only keep labels for valid images
        X = np.array(X)
        y = df.iloc[valid_indices, 1:].values
        
        print(f"\n{dataset_type} dataset:")
        print(f"Successfully loaded {len(X)} images")
        print(f"Failed to load {len(df) - len(X)} images")
        
        return X, y

class TomatoDiseaseClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.class_names = [
            'Early Blight', 'Healthy', 'Late Blight', 'Leaf Miner',
            'Leaf Mold', 'Mosaic Virus', 'Septoria', 'Spider Mites',
            'Yellow Leaf Curl Virus'
        ]
    
    def train(self, X_train, y_train, X_valid, y_valid):
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Validate model
        valid_accuracy = self.evaluate(X_valid, y_valid, "Validation")
        return valid_accuracy
    
    def evaluate(self, X, y, dataset_name="Test"):
        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        print(f"\n{dataset_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, predictions, target_names=self.class_names))
        
        return accuracy
    
    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path):
        import joblib
        classifier = cls()
        classifier.model = joblib.load(path)
        return classifier

def main():
    # Setup paths
    base_path = Path(__file__).parent
    
    # Initialize data loader
    data_loader = DataLoader(base_path)
    
    # Load datasets
    print("Loading datasets...")
    X_train, y_train = data_loader.load_dataset('train')
    X_valid, y_valid = data_loader.load_dataset('valid')
    X_test, y_test = data_loader.load_dataset('test')
    
    print(f"Dataset sizes:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Validation: {X_valid.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Initialize and train model
    classifier = TomatoDiseaseClassifier()
    valid_accuracy = classifier.train(X_train, y_train, X_valid, y_valid)
    
    # Evaluate on test set
    test_accuracy = classifier.evaluate(X_test, y_test, "Test")
    
    # Save the model
    model_path = base_path / 'models' / 'random_forest_model.joblib'
    model_path.parent.mkdir(exist_ok=True)
    classifier.save_model(model_path)
    
    return classifier, valid_accuracy, test_accuracy

if __name__ == "__main__":
    classifier, valid_accuracy, test_accuracy = main()