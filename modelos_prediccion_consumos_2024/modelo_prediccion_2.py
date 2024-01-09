import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

# Cargar el dataset
df = pd.read_excel("C:/Users/jhuamanciza/Desktop/Temporal/modelo_planificacion/dataset/dataset_muestra.xlsx")

# Convertir la columna "fecha" a formato datetime
df['fecha'] = pd.to_datetime(df['fecha'])

# Crear una columna para representar el mes y el año por separado
df['mes'] = df['fecha'].dt.month
df['anio'] = df['fecha'].dt.year

# Codificar el identificador del insumo usando LabelEncoder
le = LabelEncoder()
df['id_encoded'] = le.fit_transform(df['id'])

# Separar el dataset por insumo
insumos = df['id_encoded'].unique()

# Número de meses para la predicción
num_meses_prediccion = 6

# Lista para almacenar los resultados
resultados = []

# Entrenar y predecir para cada insumo
for insumo in insumos:
    insumo_df = df[df['id_encoded'] == insumo]

    # Renombrar columnas según la convención de Prophet
    train_data = insumo_df[['fecha', 'consumo']].rename(columns={'fecha': 'ds', 'consumo': 'y'})

    # Crear y entrenar el modelo Prophet
    model = Prophet()
    model.fit(train_data)

    # Realizar predicciones para los siguientes 6 meses
    future = model.make_future_dataframe(periods=num_meses_prediccion, freq='M')
    forecast = model.predict(future)

    # Almacenar resultados en la lista
    insumo_original = le.inverse_transform([insumo])[0]
    for i in range(1, num_meses_prediccion + 1):
        fecha_prediccion = forecast.iloc[-i]['ds']
        prediccion_consumo = max(0, forecast.iloc[-i]['yhat'])  # Asegurar que la predicción sea no negativa
        resultados.append({'Insumo': insumo_original, 'Fecha': fecha_prediccion, 'Prediccion': prediccion_consumo})

# Crear un DataFrame a partir de la lista de resultados
resultados_df = pd.DataFrame(resultados)

# Exportar el DataFrame a un archivo Excel
resultados_df.to_excel("C:/Users/jhuamanciza/Desktop/Temporal/modelo_planificacion/dataset/resultados_prediccion_prophet.xlsx", index=False)
