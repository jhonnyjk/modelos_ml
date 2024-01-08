import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Documentacion de tensorflow https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
# Referencia https://pieriantraining.com/tensorflow-lstm-example-a-beginners-guide/
# No usar bsus como codigo de muestra
# Revisar la ram


# Instanciamos pandas y le pasamos el dataset desde el archivo excel de muestra
data_path = 'C:/Users/jhuamanciza/Desktop/Modelos/datasets/Libro3.xlsx'
df = pd.read_excel(data_path)

# Noramlizar el dataset

time_series = df['tiempo']
sales_data = df['ventas']


# Creamos secuencias para la entrada y salida del modelo
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append((seq, target))
    return sequences

valor = 90

# Declaramos un numero para las predicciones y creamos las secuencias
mean_sales = sales_data.mean()
std_sales = sales_data.std()
sales_data_normalized = (sales_data - mean_sales) / std_sales
seq_length = valor
sequences = create_sequences(sales_data_normalized, seq_length)

# Dividimos en grupos para el entrenamiento y la prueba
split_ratio = 0.5
split_idx = int(split_ratio * len(sequences))
train_sequences = sequences[:split_idx]
test_sequences = sequences[split_idx:]

X_train, y_train = np.array([seq for seq, target in train_sequences]), np.array([target for seq, target in train_sequences])
X_test, y_test = np.array([seq for seq, target in test_sequences]), np.array([target for seq, target in test_sequences])

# Construimos el modelo, considerar perdidas y añadimos parametros, ver doc de tensorflow
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(1))

# Comprobamos y compilamos el modelo, si se para la ejecucion revisar el bloque anterior
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)

# Entrenamos el modelo
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Declaramos un numero para las predicciones y inicializamos una lista para guardas las predicciones
num_days_to_predict = 90
last_sequence = X_test[-1]
predictions = []

for _ in range(num_days_to_predict):
    next_pred = model.predict(last_sequence.reshape(1, seq_length, 1))
    predictions.append(next_pred[0, 0])
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = next_pred

# Normalizamos para guardar los datos
predictions = (np.array(predictions) * std_sales) + mean_sales


# Concatenamos el dataset origina con las predicciones y graficamos
plt.figure(figsize=(12, 6))
plt.plot(time_series[-len(test_sequences):], sales_data[-len(test_sequences):], label='Ventas reales')
plt.plot(pd.date_range(time_series.iloc[-1], periods=num_days_to_predict + 1, freq='D')[1:], predictions, label='Predicciones')
plt.xlabel('Tiempo')
plt.ylabel('Consumos')
plt.legend()
plt.title('Prediccion')


# Exportamos el dataset original y la prediccion a excel
predicted_dates = pd.date_range(time_series.iloc[-1], periods=num_days_to_predict + 1, freq='D')[1:]
predicted_data = pd.DataFrame({'Fecha': predicted_dates, 'Ventas Predichas': predictions})
export_path = 'C:/Users/jhuamanciza/Downloads/predicciones_tensorflow.xlsx'
predicted_data.to_excel(export_path, index=False)

plt.show()