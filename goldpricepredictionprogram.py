import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\gold_prices_1947_2026_INR.csv")

df = df[['Year', 'Gold_Price_INR_per_oz']]

df.dropna(inplace=True)

X = df[['Year']]
y = df['Gold_Price_INR_per_oz']



# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)

# Polynomial Regression
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)


year = int(input("Enter the year for gold price prediction: "))

input_year = np.array([[year]])

linear_pred = linear_model.predict(input_year)
poly_input = poly.transform(input_year)
poly_pred = poly_model.predict(poly_input)


print("\n🔮 Predicted Gold Price:")
print(f"Year: {year}")
print(f"Linear Regression Prediction: {linear_pred[0]:.2f} INR per oz")
print(f"Polynomial Regression Prediction: {poly_pred[0]:.2f} INR per oz")

# GRAPH
plt.scatter(X, y, label='Actual Data')

plt.plot(X, linear_model.predict(X), label='Linear')
plt.plot(X, poly_model.predict(X_poly), label='Polynomial')

plt.scatter(year, poly_pred[0], label='Predicted Point', marker='x', s=100)

plt.xlabel("Year")
plt.ylabel("Gold Price (INR per oz)")
plt.title("Gold Price Prediction")
plt.legend()
plt.show()