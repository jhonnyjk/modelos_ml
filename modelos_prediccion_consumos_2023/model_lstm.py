# Importa las bibliotecas necesarias
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Carga el dataset desde el archivo Excel
df = pd.read_excel("C:/Users/jhuamanciza/Desktop/Modelos/datasets/dataset.xlsx")  # Reemplaza "tu_archivo.xlsx" con la ruta de tu archivo

# Preprocesa los datos
# Asegúrate de que la columna "tiempo" esté en formato de fecha
df['tiempo'] = pd.to_datetime(df['tiempo'])
# Ordena el DataFrame por "tiempo" para garantizar el orden temporal
df = df.sort_values(by='tiempo')

# Crear una lista de identificadores únicos de productos
productos_ids = df['id'].unique()

# Crear un modelo para cada producto
modelos = {}

for producto_id in productos_ids:
    # Filtra el DataFrame para un producto en particular
    producto_df = df[df['id'] == producto_id]
    
    # Prepara los datos para el entrenamiento del modelo
    X = (producto_df['tiempo'] - producto_df['tiempo'].min()).dt.days.values  # Convierte las fechas a días desde la fecha mínima
    y = producto_df['ventas'].values
    
    # Divide los datos en entrenamiento y prueba
    split_index = int(len(X) * 0.8)  # 80% para entrenamiento, 20% para prueba
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Reshape de los datos para que sean compatibles con LSTM
    X_train = X_train.reshape(-1, 1, 1)  # (muestras, pasos de tiempo, características)
    X_test = X_test.reshape(-1, 1, 1)
    
    # Define el modelo de red neuronal con LSTM
    model = keras.Sequential([
        layers.LSTM(64, activation='relu', input_shape=(1, 1)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Una neurona de salida para las ventas
    ])
    
    # Compila el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Entrena el modelo
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    # Evalúa el modelo en los datos de prueba (opcional)
    test_loss = model.evaluate(X_test, y_test)
    print(f"Producto {producto_id} - Pérdida en datos de prueba: {test_loss}")
    
    # Guarda el modelo entrenado en el diccionario de modelos
    modelos[producto_id] = model

# Realiza predicciones para los próximos 30 días para cada producto
predicciones_futuras = {}

for producto_id in productos_ids:
    # Obtén el último punto de datos conocido para el producto
    ultimo_tiempo_conocido = df[df['id'] == producto_id]['tiempo'].max()
    
    # Crea un rango de fechas para los próximos 30 días
    fechas_futuras = pd.date_range(start=ultimo_tiempo_conocido + pd.DateOffset(1), 
                                  periods=30, 
                                  freq='D')
    
    # Convierte las fechas futuras a días desde la fecha mínima
    x_pred = (fechas_futuras - df['tiempo'].min()).days.values
    
    # Reshape de los datos para que sean compatibles con LSTM
    x_pred = x_pred.reshape(-1, 1, 1)
    
    # Realiza las predicciones
    y_pred = modelos[producto_id].predict(x_pred)
    
    # Almacena las predicciones en el diccionario
    predicciones_futuras[producto_id] = (fechas_futuras, y_pred)

# Importa la biblioteca adicional
import pandas as pd

# Crea un DataFrame para almacenar las predicciones
predicciones_df = pd.DataFrame(columns=["id", "fecha", "prediccion"])

# Llena el DataFrame con las predicciones para cada producto
for producto_id in productos_ids:
    fechas, y_pred = predicciones_futuras[producto_id]
    df_temp = pd.DataFrame({
        "id": [producto_id] * len(fechas),
        "fecha": fechas,
        "prediccion": y_pred.flatten()
    })
    predicciones_df = pd.concat([predicciones_df, df_temp], ignore_index=True)

# Guarda el DataFrame en un archivo Excel
predicciones_df.to_excel("C:/Users/jhuamanciza/Downloads/predicciones_tensorflow.xlsx", index=False)  # Cambia el nombre del archivo según tu preferencia
