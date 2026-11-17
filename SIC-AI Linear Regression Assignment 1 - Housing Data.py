import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

housing_df = pd.read_csv("Housing.csv")
feature = housing_df.iloc[:,1:].values
output = housing_df.iloc[:,0].values

categorical_indices = [4, 5, 6, 7, 8, 10, 11]
encoders = {}
for index in categorical_indices:
    en = LabelEncoder()
    feature[:,index] = en.fit_transform(feature[:,index])
    encoders[index] = en

sc = StandardScaler()
feature_scaled = sc.fit_transform(feature)

x_train, x_test, y_train, y_test = train_test_split(feature_scaled, output, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Mean Absolute Error = ", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error = ", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error = ", mean_squared_error(y_test, y_pred)**0.5)
print("R2 Score = ", r2_score(y_test, y_pred))

predict = np.array([[1300, 3, 1, 2, 'no', 'no', 'yes', 'yes', 'yes', 0, 'no', 'semi-furnished']], dtype=object)

for idx in categorical_indices:
    predict[:,idx] = encoders[idx].transform(predict[:,idx])

predict_scaled = sc.fit_transform(predict.astype(float))

predicted_price = model.predict(predict_scaled)
print("Predicted House Price: ", predicted_price[0])


sns.set_theme(style="whitegrid")
print("--- Numerical Summary Statistics ---")
print(housing_df.describe())

# Distribution of the Target Variable (Price)
plt.figure(figsize=(8, 4))
sns.histplot(housing_df['price'], kde=True, color='blue')
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()

# Correlation Heatmap for Numerical Features
plt.figure(figsize=(10, 6))
numerical_cols = housing_df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = housing_df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Area vs Price (Colored by Air Conditioning)
plt.figure(figsize=(9, 5))
sns.scatterplot(data=housing_df, x='area', y='price', hue='airconditioning', palette='Set1', alpha=0.7)
plt.title('House Price vs. Area (Categorized by AC)')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.show()

# Furnishing Status vs Price (Box Plot)
plt.figure(figsize=(8, 5))
sns.boxplot(data=housing_df, x='furnishingstatus', y='price', palette='Set2', hue='furnishingstatus', legend=False)
plt.title('Price Distribution by Furnishing Status')
plt.xlabel('Furnishing Status')
plt.ylabel('Price')
plt.show()