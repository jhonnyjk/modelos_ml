import pandas as pd
import numpy as np

# Cargar el conjunto de datos desde el archivo Excel
df = pd.read_excel('C:/Users/jhuamanciza/Desktop/Temporal/modelo_planificacion/dataset/dataset.xlsx')  # Asegúrate de proporcionar la ruta correcta

# Convertir la columna 'tiempo' a formato datetime si no lo está
df['tiempo'] = pd.to_datetime(df['tiempo'])

# Obtener la lista de todos los SKUs en el conjunto de datos
skus = df['material'].unique()

# Definir el parámetro k
k = 5  # Puedes ajustar este valor según tus necesidades

# Diccionario para almacenar los resultados por SKU
resultados_por_sku = {}

# Iterar sobre cada SKU
for sku in skus:
    # Filtrar el conjunto de datos para el SKU actual
    df_sku = df[df['material'] == sku]

    # Obtener los últimos 12 meses para estimar la estacionalidad
    consumos_ultimos_12 = df_sku['consumo'].tail(12).tolist()

    # Tomar los k-ésimos mayores
    mayores_consumos = np.partition(consumos_ultimos_12, -k)[-k:]

    # Calcular el rango de cobertura en unidades
    cobertura_min = int(np.percentile(mayores_consumos, 10))  # Percentil 10
    cobertura_max = int(np.percentile(mayores_consumos, 90))  # Percentil 90

    # Calcular el consumo promedio
    consumo_promedio = np.mean(consumos_ultimos_12)

    # Calcular el rango de cobertura en días
    dias_por_mes = 30  # Supongamos que un mes tiene 30 días (ajusta según tu caso)
    cobertura_min_dias = cobertura_min / consumo_promedio * dias_por_mes
    cobertura_max_dias = cobertura_max / consumo_promedio * dias_por_mes

    # Almacenar los resultados en el diccionario
    resultados_por_sku[sku] = {
        'cobertura_min': cobertura_min,
        'cobertura_max': cobertura_max,
        'cobertura_min_dias': cobertura_min_dias,
        'cobertura_max_dias': cobertura_max_dias
    }

# Crear un DataFrame a partir de los resultados
df_resultados = pd.DataFrame.from_dict(resultados_por_sku, orient='index')

# Exportar el DataFrame a un archivo Excel en la misma carpeta del dataset
df_resultados.to_excel('resultados_cobertura.xlsx', index_label='SKU')

# Imprimir mensaje de éxito
print("Resultados exportados correctamente a 'resultados_cobertura.xlsx'")
