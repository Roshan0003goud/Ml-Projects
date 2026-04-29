import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load data
df = pd.read_csv("tested.csv")

# Step 2: Fix ALL missing values (new pandas way)
df = df.copy()  # fix chained assignment warning

# Fill missing numbers
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Remove Cabin column (too many missing)
df = df.drop(columns=["Cabin"])

# Convert Sex to numbers
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Remove any remaining missing rows
df = df.dropna()

# Confirm no missing values
print("Missing values after fixing:")
print(df.isnull().sum())
print(f"Total rows remaining: {len(df)}")

# Step 3: Pick columns
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train model
model = LogisticRegression(
    solver="liblinear",
    max_iter=1000
)
model.fit(X_train, y_train)

# Step 6: Check accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")