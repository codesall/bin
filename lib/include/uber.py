#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings 
#We do not want to see warnings
warnings.filterwarnings("ignore") 

#import data
data = pd.read_csv("uber.csv")

#Create a data copy
df = data.copy()

#Print data
df.head()

#Get Info
df.info()

#pickup_datetime is not in required data format
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

df.info()

#Statistics of data
df.describe()

#Number of missing values
df.isnull().sum()

#Correlation
df.select_dtypes(include=[np.number]).corr()

print(df.columns)

#Drop the rows with missing values
df.dropna(inplace=True)

plt.boxplot(df['fare_amount'])

#Remove Outliers
q_low = df["fare_amount"].quantile(0.01)
q_hi  = df["fare_amount"].quantile(0.99)

df = df[(df["fare_amount"] < q_hi) & (df["fare_amount"] > q_low)]

#Check the missing values now
df.isnull().sum()

#Time to apply learning models
from sklearn.model_selection import train_test_split

#Take x as predictor variables and y as response variable
x = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
y = df['fare_amount']

#Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)
y_pred_lm = lm.predict(x_test)

#Check R2 score
from sklearn.metrics import r2_score, mean_squared_error
r2_lm = r2_score(y_test, y_pred_lm)
rmse_lm = np.sqrt(mean_squared_error(y_test, y_pred_lm))

print("Linear Regression Results:")
print(f"R2 Score: {r2_lm}")
print(f"RMSE: {rmse_lm}")

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

#Evaluate Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\nRandom Forest Results:")
print(f"R2 Score: {r2_rf}")
print(f"RMSE: {rmse_rf}")

#Comparison
print("\nModel Comparison:")
print(f"Linear Regression - R2: {r2_lm:.4f}, RMSE: {rmse_lm:.4f}")
print(f"Random Forest      - R2: {r2_rf:.4f}, RMSE: {rmse_rf:.4f}")

#Prediction function
def predict_fare(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count):
    test_df = pd.DataFrame({
        'pickup_longitude': [pickup_longitude],
        'pickup_latitude': [pickup_latitude],
        'dropoff_longitude': [dropoff_longitude],
        'dropoff_latitude': [dropoff_latitude],
        'passenger_count': [passenger_count]
    })
    
    fare_lm = lm.predict(test_df)[0]
    fare_rf = rf.predict(test_df)[0]
    
    print("\nPredicted Fare (Linear Regression):", round(fare_lm, 2))
    print("Predicted Fare (Random Forest):", round(fare_rf, 2))

#Example prediction
predict_fare(-73.95, 40.78, -73.98, 40.76, 1)
