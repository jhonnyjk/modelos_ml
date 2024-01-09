
# Predicción de consumos


Este repositorio contiene un script en Python para predecir el consumo de insumos utilizando un modelo RandomForestRegressor. El código está diseñado para trabajar con un conjunto de datos que contiene información temporal sobre el consumo de diferentes insumos.

## Requisitos
Asegúrate de tener instalado Python ^3.10 y las siguientes bibliotecas:
```
pip install pandas 
pip install scikit-learn
pip install datetime

```
## Uso

1. Clona este repositorio:
```
https://github.com/jhonnyjk/modelos_ml.git
```
2. Modifica la ruta de donde se cargara el dataset
3. Modifica la ruta de donde se exportara el dataframe con con las predicciones
4. Ejecuta el script de Python:
```Python
python prediccion_consumo.py
```
El script cargará el conjunto de datos, entrenará un modelo RandomForestRegressor para cada insumo, realizará predicciones para los próximos 6 meses y exportará los resultados a un archivo Excel.

## Configuración del Modelo
El modelo RandomForestRegressor se configura con 100 estimadores y una semilla aleatoria de 42. Puedes ajustar estos parámetros según tus necesidades modificando el script.
```Python
model = RandomForestRegressor(n_estimators=100, random_state=42)
```
Puedes modificar la cantidad de meses a predecir
```Python
num_meses_prediccion = 6
```

### Resultados
Los resultados de las predicciones se almacenan en un archivo Excel llamado resultados_prediccion.xlsx en la ruta especificada en el paso 3.

### Contribuciones
¡Las contribuciones son bienvenidas! Si encuentras algún problema o tienes ideas para mejorar el código, no dudes en abrir un problema o enviar una solicitud de extracción.

### Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para obtener más detalles.