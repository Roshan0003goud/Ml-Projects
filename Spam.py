import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading File
df = pd.read_csv("spam.csv", encoding="latin-1")

# use useful columns only
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# convert ham/sam into numbers
df["label"] = df["label"].map(
    {"ham": 0, "spam": 1}
)

#converting text  into numbers 

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["message"])
y = df["label"]

# spilt data 
X_train, X_test, y_train, y_test, = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = LogisticRegression(
    solver="liblinear",
    max_iter=1000
)
model.fit(X_train, y_train)

# check accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy:{accuracy * 100:.2f}%")
