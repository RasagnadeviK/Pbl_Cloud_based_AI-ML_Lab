import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generating some sample data
np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 100)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 100)

# Transforming the data to include another axis to fit the sklearn requirement
x = x[:, np.newaxis]
y = y[:, np.newaxis]

# Polynomial features
poly_features = PolynomialFeatures(degree=3)
x_poly = poly_features.fit_transform(x)

# Fit the model
model = LinearRegression()
model.fit(x_poly, y)

# Predict
y_pred = model.predict(x_poly)

# Plotting the actual data and the polynomial regression line
plt.scatter(x, y, s=10, label='Actual data')
plt.plot(x, y_pred, color='r', label='Polynomial regression')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Polynomial Linear Regression')
plt.legend()
plt.show()
