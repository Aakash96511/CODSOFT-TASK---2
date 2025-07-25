# Movie Rating Prediction using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("IMDb-India.csv")

# Encode categorical features
label = LabelEncoder()
df['Genre'] = label.fit_transform(df['Genre'])
df['Director'] = label.fit_transform(df['Director'])
df['Actors'] = label.fit_transform(df['Actors'])

# Features and target
X = df[['Genre', 'Director', 'Actors', 'Runtime']]
y = df['Rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("✅ Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("✅ R² Score:", r2_score(y_test, y_pred))
