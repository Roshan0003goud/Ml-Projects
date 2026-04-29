import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load data
df = pd.read_csv("IRIS.csv")

#step2: Explore
print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

# Columns Pick
X= df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df[["species"]]

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training Model
model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)
model.fit(X_train, y_train)

# Step 6: Check accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")