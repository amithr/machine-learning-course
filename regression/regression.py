# https://thecleverprogrammer.com/2022/10/31/salary-prediction-with-machine-learning/

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
'''
pip install pandas
pip install numpy
pip install plotly
pip install scikit-learn
'''




data = pd.read_csv("regression/salary_data.csv")
#print(data.head())

# Visualize relationship between salary and job experience
figure = px.scatter(data_frame = data, 
                    x="Salary",
                    y="YearsExperience", 
                    size="YearsExperience", 
                    trendline="ols")
# figure.show()

# Split into training and testing set
x = np.asanyarray(data[["YearsExperience"]])
y = np.asanyarray(data[["Salary"]])
# test_size is the proportion of the data that will be the test dataset
# It is a seed for a random number generator that decides which values will be chosen
# for the test dataset
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

# xtest 
# Train the machine learning model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Make predictions on the test set
y_pred = model.predict(xtest)

# Evaluate the model's performance on the test set
# mse measured the average of the squared differences between the actual and predicted values
mse = mean_squared_error(ytest, y_pred)
# Measure of how well a model explains variation in dependent variable
# Higher r2 means a more accurate model
r2 = r2_score(ytest, y_pred)

# Make a prediction
a = float(input("Years of Experience : "))
features = np.array([[a]])
print("Predicted Salary = ", model.predict(features))
