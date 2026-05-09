import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("titanic/tested.csv")

# clean the data
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# dropping columns
df = df.drop(columns=["Cabin"])

# convert sex into numbers
df["Sex"] = df["Sex"].map({"male":0, "female":1})

# New Feature1 : Is child ?

df["Is_Child"] = (df["Age"] < 18).astype(int)

# New Fature : Is Alone ?
df["Is_Alone"] = (
    (df["SibSp"] == 0) &
    (df["Parch"] == 0)
).astype(int)

# New Feature : Family Size

df["Family_Size"] = df["SibSp"] + df["Parch"]

# See New Columns

print("New Features created:")
print(df[["Age", "Is_Child",
          "Is_Alone",
          "Family_Size"]].head(10))



y = df["Survived"]

# Model 1 : Without Feature Engineering
X_basic = df[["Pclass", "Sex", "Age", "Fare"]]

X_train, X_test, y_train, y_test = train_test_split(
    X_basic, y, test_size=0.2, random_state=42
)

# training model

model1 = LogisticRegression(
    solver="liblinear",
    max_iter=1000
)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)
acc1 = accuracy_score(y_test, pred1)
print(f"\nWithout feature engineering:{acc1 * 100: .2f}%")

# Model 2 : with Feature Engineering 
X_Advanced = df[["Pclass", "Sex", "Age",
                 "Fare", "Is_Child",
                 "Is_Alone", "Family_Size"]]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_Advanced, y, test_size=0.2,
    random_state=42
)

# Training model
model2 = LogisticRegression(
    solver="liblinear",
    max_iter=1000
)
model2.fit(X_train2, y_train2)

pred2 = model2.predict(X_test2)
acc2 = accuracy_score(y_test2, pred2)
print(f"\n With Feature Engineering:{acc2 *100: .2f}%")

# Compare

print(f"\n Compare both :{(acc2 - acc1) * 100: .2f}%")
