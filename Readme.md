# Titanic Survival Prediction 🚢

## Project Overview
This project predicts whether a passenger 
survived the Titanic disaster using 
Machine Learning.

## What I Did
- Loaded the Titanic dataset
- Explored and cleaned the data
- Built a Machine Learning model
- Got 100% accuracy!

## Dataset
- Source: Kaggle
- File: tested.csv
- Total rows: 418 passengers
- Columns: PassengerId, Survived, Pclass,
  Name, Sex, Age, SibSp, Parch, 
  Ticket, Fare, Cabin, Embarked

## Problems Found in Data
| Column | Missing Values |
|--------|---------------|
| Age    | 86 missing    |
| Fare   | 1 missing     |
| Cabin  | 327 missing   |

## How I Fixed It
- Age → Filled with median value
- Fare → Filled with median value
- Cabin → Dropped (too many missing!)
- Sex → Converted male/female to 0/1

## Model Used
- Algorithm: Logistic Regression
- Library: scikit-learn
- Train/Test Split: 80% / 20%

## Results
- Model Accuracy: 100% ✅

## What I Learned
- How to load and explore CSV data
- How to find and fix missing values
- Why we use median instead of mean
- How to convert text to numbers
- How to split data into train/test
- How to build ML model
- How to measure accuracy

## Tools Used
- Python 3
- pandas
- scikit-learn
- VS Code

## How to Run
1. Install required libraries:
pip install pandas scikit-learn

2. Run the file:
python titanic.py

## Author
Roshan
GitHub: https://github.com/Roshan0003goud