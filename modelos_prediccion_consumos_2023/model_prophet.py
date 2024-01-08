import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Documentacion de prophet https://facebook.github.io/prophet/docs/quick_start.html#r-api
# Estacionalidad y hiperparametros para ajuster el modelo $probar https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html

# Instanciamos pandas para usar sus metodos y cargamos el dataset desde el excel de muestra
data = pd.read_excel('C:/Users/jhuamanciza/Desktop/Modelos/datasets/Libro3.xlsx')

# Renombramos las columnas a 'ds' y 'y' para que prophet lo pueda recibir
data = data.rename(columns={'tiempo': 'ds', 'ventas': 'y'})

# Instanciamos prophet y le pasamos el dataset al metodo fit para que puedan ser ajustados
model = Prophet()
model.fit(data)

# Dataframe con fechas para las predicciones
future = model.make_future_dataframe(periods=54)
forecast = model.predict(future)

# Creamos el grafico y concatenamos
fig, ax = plt.subplots(figsize=(12, 6))
model.plot(forecast, ax=ax)
ax.set_xlabel('Fecha')
ax.set_ylabel('Consumos')
ax.set_title('Prediccion')

# Exportamos las predicciones
def to_excel(predict, path_excel):
    predict.to_excel(path_excel, index=False)
path_excel = 'C:/Users/jhuamanciza/Downloads/predicciones_prophet.xlsx'
to_excel(forecast, path_excel)

# Graficowe
plt.show()