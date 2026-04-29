import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
df = pd.read_csv("house_price_regression_dataset.csv")

print(df.head())

print(df.info())

print(df.isnull().sum())

print(df.duplicated().sum())

# Pick Colmuns
X = df[["Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality"]]
y = df["House_Price"]

#  Spiltting data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# train Model

model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained!")

# Step 6: Test model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"\nAverage Error: ${mae:,.2f}")