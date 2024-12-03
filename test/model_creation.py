import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save a small test set to CSV
X_test.to_csv("iris_sample.csv", index=False)
print("Sample data saved as 'iris_sample.csv'")


# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model to a file
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'iris_model.pkl'")
