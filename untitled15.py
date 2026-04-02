# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

API_KEY ='4a4b1efebeaf8ca67c8a72a3d7d31521'
BASE_URL = 'http://api.openweathermap.org/data/2.5/'

def get_current_weather(city):
  url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
  response = requests.get(url)
  data = response.json()
  return{

       'city': data['name'],
       'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
       'temp_max': round(data['main']['temp_max']),
       'humidity': round(data['main']['humidity']),
       'description': data['weather'][0]['description'],
       'country': data['sys']['country'],
       'Wind_gust_dir': data['wind']['deg'],
       'pressure': data['main']['pressure'],
        'Wind_Gust_Speed': data['wind']['speed']

  }

def read_histrorical_data(filename):
  df= pd.read_csv(filename)
  df = df.dropna()
  df=df.drop_duplicates()
  return df

def prepare_data(data):
  le= LabelEncoder()
  data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
  data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
  X=  data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']]
  Y= data['RainTomorrow']
  return X, Y, le

def train_rain_model(X,y):
    X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(" Mean Squared Error for Rain Model")
    print(mean_squared_error(y_test,y_pred))
    return model

def prepare_regression_data(data,feature):
  X,y =[],[]
  for i in range(len(data)-1):
    X.append(data[feature].iloc[i])

    y.append(data[feature].iloc[i+1])

    X = np.array(X).reshape(-1,1)
    y=np.array(y)
    return X,y

def predict_future(model, current_value, S=7):
  prediction = [current_value]
  for i in range(5):
    next_value = model.predict(np.array([[prediction[-1]]]))[0]
    prediction.append(next_value)
  return prediction[1:]

def weather_view():
  city= input('Enter any city name:')
  current_weather = get_current_weather(city)
  try:
    histroical_data = read_histrorical_data('/content/weather.csv')
  except FileNotFoundError:
    print("Error: The historical data file 'weather.csv' was not found at '/content/weather.csv'.")
    print("Please upload 'weather.csv' to your Colab environment or ensure it is in the correct path.")
    return # Exit the function if the file is not found
  X,y,le = prepare_data(histroical_data)
  rain_model = train_rain_model(X,y)
  Wind_deg = current_weather['Wind_gust_dir']%360
  compass_point = [
      ("N",0,11.25),("NNE",11.25,33.75),("NE",33.75,56.25),("ENE",56.25,78.75),
      ("E",78.75,101.25),("ESE",101.25,123.75),("SE",123.75,146.25),("SSE",146.25,168.75),
      ("S",168.75,191.25),("SSW",191.25,213.75),("SW",213.75,236.25),("WSW",236.25,258.75),
      ("W",258.75,281.25),("WNW",281.25,303.75),("NW",303.75,326.25),("NNW",326.25,348.75)
  ]

  compass_direction= next(point for point, start,end in compass_point if start<= Wind_deg<end)
  compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1
  current_data = {
        'MinTemp': current_weather['temp_min'],
       'MaxTemp': current_weather['temp_max'],
      'WindGustDir': compass_direction_encoded,
      'WindGustSpeed': current_weather['Wind_Gust_Speed'],
      'Humidity': current_weather['humidity'],
      'Pressure': current_weather['pressure'],
      'Temp': current_weather['current_temp'],
  }

  Current_df = pd.DataFrame([current_data])
  rain_prediction = rain_model.predict(Current_df)[0]

  X_temp, y_temp = prepare_regression_data(histroical_data, 'Temp')

  X_hum, y_hum = prepare_regression_data(histroical_data, 'Humidity')

  temp_model = train_regression_model(X_temp, y_temp)

  hum_model = train_regression_model(X_hum, y_hum)

  future_temp = predict_future(temp_model, current_weather['temp_min'] )
  future_humidity = predict_future(hum_model, current_weather['humidity'])
  import pytz
  timezone = pytz.timezone('Asia/Kolkata')
  now = datetime.now(timezone)
  next_hours = now + timedelta(hours=1)
  next_hours= next_hours.replace(minute=0,second=0,microsecond=0)
  future_times=[(next_hours + timedelta(hours=i)).strftime("%H:00") for i in range(10)]

  print(f"City: {city},{current_weather['country']}")
  print(f"CurrentTemperature: {current_weather['current_temp']}c")
  print(f"Feels like: {current_weather['feels_like']}c")
  print(f"Humidity: {current_weather['humidity']}%")
  print(f" Minimum Temperature: {current_weather['temp_min']}c")
  print(f" Maxmium Temperature: {current_weather['temp_max']}c")
  print(f" Rain prediction: {'YES' if rain_prediction else 'NO'}")
  print("\nFuture Temperature Predictions:")

  for time,temp in zip(future_times,future_temp):
    print(f" {time}:{round(temp, 1)}c")
  print("\nFuture Humidity Predictions:")

  for time,hum_value in zip(future_times,future_humidity):
    print(f"  {time}{round(hum_value,1)}%")
weather_view()
