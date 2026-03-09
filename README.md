# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula.

## Program:
```
#Ex 09 - Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant',
                 'Manager', 'Country Manager', 'Region Manager',
                 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Split features and target
# ------------------------------
X = df[['Level']]     # Feature (Level)
y = df['Salary']      # Target (Salary)

# ------------------------------
# Step 3: Create Decision Tree Regressor
# ------------------------------
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

# ------------------------------
# Step 4: Predict salary for the dataset or new levels
# ------------------------------
y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)

# Example: predict salary for a new employee at level 6.5
level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")

# ------------------------------
# Step 5: Visualize the results (High-resolution curve)
# ------------------------------
X_grid = np.arange(min(X.values), max(X.values)+0.01, 0.01)  # High-resolution for smoother curve
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: B ARPUTHA
RegisterNumber:  25012532
*/
```

## Output:
<img width="920" height="71" alt="image" src="https://github.com/user-attachments/assets/f1a1c301-50cd-48f2-9ad3-c24e9aac977f" />
<img width="699" height="565" alt="image" src="https://github.com/user-attachments/assets/41355c11-c9c8-4f67-8463-d008afb63607" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
