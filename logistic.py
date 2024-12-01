import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import random as rn

# Load dataset
data = pd.read_csv(r'D:\Shiv\Stars.csv')

# Print dataset columns
print("Columns in the dataset:", data.columns)
# Drop the columns that are not used in this analysis
data = data.drop(columns=['Color','Spectral_Class'])
# Define features and  target variable
X = data[['Temperature', 'L', 'R', 'A_M']]
Y = data['Type']  # Target variable

# Split the dataset into training and testing sets
rn.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the logistic regression model with hyperparameter tuning
model = LogisticRegression(solver='liblinear')  # Solver for small datasets
param_grid = {'C': [0.1, 1, 10, 100]}  # Regularization parameter grid
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Train the model with the best parameters
grid_search.fit(X_train_scaled, Y_train)

# Make predictions
Y_pred = grid_search.best_estimator_.predict(X_test_scaled)

# Print accuracy score
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display the best parameters found by GridSearchCV
print("Best parameters:", grid_search.best_params_)
