import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def clasificar_rotacion_y_demanda(dataset_path):
    # Cargar el dataset desde el archivo Excel
    df = pd.read_excel(dataset_path)

    # Obtener la lista única de materiales (IDs de SKU)
    skus = df['material'].unique()

    # Inicializar listas para almacenar los resultados de cada SKU
    resultados_por_sku = []

    for sku in skus:
        # Filtrar el dataset para el SKU actual
        sku_data = df[df['material'] == sku]

        # Obtener la serie temporal y consumos
        tiempo = sku_data['tiempo'].values
        consumos = sku_data['consumo'].values

        clasificacion = {}
        demanda_tipo = {}

        meses_con_consumo = np.count_nonzero(consumos)
        proporciones_meses_con_consumo = meses_con_consumo / len(consumos)

        # Lógica de Clasificación
        if proporciones_meses_con_consumo >= 0.75:
            clasificacion['Clasificación'] = 'Alta Rotación'
        elif 0.25 <= proporciones_meses_con_consumo < 0.75:
            clasificacion['Clasificación'] = 'Baja Rotación'
        else:
            clasificacion['Clasificación'] = 'Uso Inmediato'

        # Cálculo del Coeficiente de Variación (CV)
        cv = np.std(consumos) / np.mean(consumos)
        cov_squared = cv ** 2

        # Cálculo de la ADI (Average Demand Interval)
        adi = len(consumos) / meses_con_consumo if meses_con_consumo > 0 else np.nan

        # Lógica de Tipo de Demanda
        if adi < 1.32 and cov_squared < 0.49:
            demanda_tipo['Tipo de Demanda'] = 'Suavizado (Smooth demand)'
        elif adi >= 1.32 and cov_squared < 0.49:
            demanda_tipo['Tipo de Demanda'] = 'Intermitente (Intermittent demand)'
        elif adi < 1.32 and cov_squared >= 0.49:
            demanda_tipo['Tipo de Demanda'] = 'Errática (Erratic demand)'
        elif adi >= 1.32 and cov_squared >= 0.49:
            demanda_tipo['Tipo de Demanda'] = 'Grumosa (Lumpy demand)'

        # Crear un DataFrame con los resultados para el SKU actual
        resultados_sku = pd.DataFrame({
            'SKU': [sku],
            'Clasificación': [clasificacion['Clasificación']],
            'Proporción Meses con Consumo': [proporciones_meses_con_consumo],
            'Coeficiente de Variación (CV)': [cv],
            'ADI (Average Demand Interval)': [adi],
            'Tipo de Demanda': [demanda_tipo['Tipo de Demanda']]
        })

        # Imprimir el cuadro resumen para el SKU actual
        print(f"Resultados para SKU {sku}:")
        print(resultados_sku.to_string(index=False))
        print()

        # Agregar los resultados del SKU actual a la lista general
        resultados_por_sku.append(resultados_sku)

    # Concatenar los resultados de todos los SKU en un solo DataFrame
    resultados_totales = pd.concat(resultados_por_sku, ignore_index=True)

    # Exportar los resultados totales a un archivo Excel en la misma carpeta
    export_path = os.path.join(os.path.dirname(dataset_path), 'resultados_demanda_por_sku.xlsx')
    resultados_totales.to_excel(export_path, index=False)
    print(f"Resultados totales exportados a: {export_path}")

    return resultados_totales

# Ruta al archivo Excel con el dataset
dataset_path = 'C:/Users/jhuamanciza/Desktop/Temporal/modelo_planificacion/dataset/dataset.xlsx'

# Obtener la clasificación y análisis para el dataset
resultados_totales = clasificar_rotacion_y_demanda(dataset_path)
