import pandas as pd
#import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv(r'D:\Shiv\Stars.csv')

# Drop the 'Color' column
data = data.drop(columns=['Color','Spectral_Class'])

# Encode the 'Type' column if it contains strings

# Display a scatterplot of the dataset
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data)
plt.title('Scatterplot of the Stars Dataset')
plt.show()

# Define features and target variable
X = data[['Temperature', 'L', 'R', 'A_M']]  # Features
Y = data['Type']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)

# Evaluate the model using MSE and R² Score
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score (Accuracy): {r2 * 100:.2f}%")
