import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load a CSV file
df = pd.read_csv("churn.csv")


#Churn - converting churn into numbers
df["Churn"] = df["Churn"].map({"Yes":1, "No":2}) 


# contract - converting text into numbers
df["Contract"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})

# convert - Total charges into numbers 
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())


# selecting the columns
X = df[["tenure", "MonthlyCharges", "TotalCharges", "Contract"]]
y = df["Churn"]

# Splitting data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

# train model
model = LogisticRegression(
    solver="liblinear"
)
model.fit(X_train, y_train)

# check accuracy
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print(f"Model Accuracy:{accuracy * 100:.2f}%")
