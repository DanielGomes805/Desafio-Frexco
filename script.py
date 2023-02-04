import pandas as pd
import numpy as np
from fbprophet import Prophet

# Carregando o dataset
df = pd.read_csv("https://docs.google.com/spreadsheets/d/1dybG-YD9AoXj9XhRbHHRI-chMl-SlSRC/edit?usp=sharing&ouid=102272984259281951520&rtpof=true&sd=true")

# Renomeando as colunas para o padrão do Prophet
df = df.rename(columns={"date": "ds", "demand": "y"})

# Treinando o modelo Prophet
model = Prophet()
model.fit(df)

# Criando a previsão para os próximos 5 dias
future = model.make_future_dataframe(periods=5)
forecast = model.predict(future)

# Extraindo a previsão de demanda
demand_forecast = forecast[['ds', 'yhat']]
print(demand_forecast)