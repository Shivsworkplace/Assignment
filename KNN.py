import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import random as rn

# Load dataset

data = pd.read_csv(r'D:\Shiv\Stars.csv')

# Drop the columns that are not used in this analysis
data = data.drop(columns=['Color','Spectral_Class'])



# Print dataset columns
print("Columns in the dataset:", data.columns)

# Define features and target variable
X = data[['Temperature', 'L', 'R', 'A_M']]  # Features (using more than one feature for better analysis)
Y = data['Type']  # Target variable

# Split the dataset into training and testing sets
rn.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

# Initialize and train the KNN model
model = KNeighborsClassifier(n_neighbors=3)  # You can adjust 'n_neighbors' as needed
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)

# Print accuracy score
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
