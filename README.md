import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the CSV file
csv_file_path = r'C:\Users\PMchannaveeradevaru\Downloads\project\1inch.csv'  # Replace with your actual CSV file name

# Load the dataset
df = pd.read_csv(csv_file_path)

# Display the first few rows
print(df.head())

# Data Preprocessing
df.fillna(method='ffill', inplace=True)

# Calculate price change and create target variable
df['Price_Change'] = df['Close'].pct_change()
df['Trade_Quality'] = np.where(df['Price_Change'] > 0, 1, 0)

# Drop rows with NaN values
df.dropna(inplace=True)

# Features and target variable
X = df[['Open', 'High', 'Low', 'Volume']]  # Example features
y = df['Trade_Quality']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot feature importance
feature_importances = model.feature_importances_
features = X.columns
indices = np.argsort(feature_importances)[::-1]

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
