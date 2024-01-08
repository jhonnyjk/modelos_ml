import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Documentacion de statsmodels con sarimax https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
# Referencia https://analyticsindiamag.com/complete-guide-to-sarimax-in-python-for-time-series-modeling/
# No usar bsus como codigo de muestra



# Instanciamos pandas y le pasame el dataset desde excel de muestra, se debe parsear y concatenar.
file_path = 'C:/Users/jhuamanciza/Desktop/Modelos/datasets/Libro3.xlsx'
df = pd.read_excel(file_path, parse_dates=['tiempo'], index_col='tiempo')

# Ajustar el modelo  con ordenes manuales p - d - q y P - D - Q - s,
p, d, q = 2, 1, 2
P, D, Q, s = 1, 1, 1, 7
sarima_model_fit = SARIMAX(df['ventas'], order=(p, d, q), seasonal_order=(P, D, Q, s)).fit()

# Declaramos un numero para las predicciones y lo pasamos como parametro para el ajuste y la prediccion
forecast_steps = 90
forecast = sarima_model_fit.get_forecast(steps=forecast_steps)

# Obtenemos los datos de la prediccion y los intervalos de confianza
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Creamos una lista de fechas para concatenar las predicciones
last_date = df.index[-1]
date_range = pd.date_range(last_date, periods=forecast_steps + 1)

# Creamos un dataframe para las predicciones 
forecast_df = pd.DataFrame({'Fecha': date_range[1:], 'Predicción': forecast_values})
forecast_df.set_index('Fecha', inplace=True)

# Concatenamos los el dataset original con las predicciones
plt.figure(figsize=(10, 6))
plt.plot(df['ventas'], label='Consumos')
plt.plot(forecast_df['Predicción'], label='Predicción')
plt.fill_between(forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='gray', alpha=0.2, label='Intervalo de confianza')
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('Consumos')
plt.title('Predicciones')


# Exportamos la predicciones
path_output_excel = 'C:/Users/jhuamanciza/Downloads/prediccion_sarima.xlsx'
forecast_df.to_excel(path_output_excel)

plt.show()
