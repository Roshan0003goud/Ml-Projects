import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("titanic/tested.csv")

print(df.isnull().sum())

# clean data
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# deleting columns 
df = df.drop(columns=["Cabin"])

# Converting str to numbers 
df["Sex"] = df["Sex"].map({"Male":0, "Female": 1})

# Selecting Columns 
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df[["Survived"]]

# Spiltting
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42
)

# Trainning model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
Accuracy = accuracy_score(y_test, predictions)

print(f"\n Decision tree Accurarcy : {Accuracy *100: .2f}%")

# To See which columns is more important 
for feature, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(f"{feature}: {importance *100:.2f}%")
