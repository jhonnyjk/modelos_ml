import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

    # Dividir el conjunto de datos en entrenamiento y prueba
    train_data, test_data = train_test_split(insumo_df, test_size=0.2, shuffle=False)

    # Seleccionar características y etiquetas
    X_train = train_data[['mes', 'anio']]
    y_train = train_data['consumo']

    # Crear y entrenar el modelo RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Realizar predicciones para los siguientes 12 meses
    last_month = insumo_df['fecha'].max()
    next_months = [last_month + pd.DateOffset(months=i) for i in range(1, num_meses_prediccion + 1)]
    next_months_data = pd.DataFrame({'mes': [m.month for m in next_months], 'anio': [m.year for m in next_months]})

    predicted_consumo = model.predict(next_months_data)
    predicted_consumo = [max(0, consumo) for consumo in predicted_consumo]  # Asegurar que la predicción sea no negativa

    # Almacenar resultados en la lista
    insumo_original = le.inverse_transform([insumo])[0]
    for i, month in enumerate(next_months):
        resultados.append({'Insumo': insumo_original, 'Fecha': month, 'Prediccion': predicted_consumo[i]})

# Crear un DataFrame a partir de la lista de resultados
resultados_df = pd.DataFrame(resultados)

# Exportar el DataFrame a un archivo Excel
resultados_df.to_excel("C:/Users/jhuamanciza/Desktop/Temporal/modelo_planificacion/dataset/resultados_prediccion.xlsx", index=False)
