import pandas as pd

#step 1: Load data
df = pd.read_csv("students.csv")

# step 2: print original data
print(df)

# step 3: explore 
print(df.head())       # first rows - by default it prints first 5 rows only

print(df.info())       # data types

print(df.describe())   # statistics

#  step 4: basic analysis

print("average math:", df["math"].mean())
print("highest science:", df["science"].max())

# step 5 : adding columns

df["total"] = df["math"] + df["science"] + df["english"]
df["average"] = df["total"] / 3
 
print(df)

# step5: find highest average student

best = df[df["average"] == df["average"].max()]
print("best student")
print(best)

# step6: find lowest average student
 
lower = df[df["average"] == df["average"].min()]
print("low student")
print(lower)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
