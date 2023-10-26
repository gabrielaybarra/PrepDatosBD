# DataProcessing
Esta librería tiene como objetivo simplificar y agilizar el proceso de preparación y limpieza de datos, lo que es fundamental en cualquier análisis de datos o proyecto de aprendizaje automático. Al proporcionar una variedad de herramientas y funciones, los usuarios pueden trabajar de manera más eficiente y garantizar la calidad de los datos con los que están trabajando.

Estas son las librerías necesarias para ejecutar las funciones de la librería:
- import pandas as pd
- import matplotlib.pyplot as plt
- import sklearn.impute import KNNImputer
- import seaborn as sns


class preprocess():
- view_nan_graph(tabla_nan)
- drop_column(df, column_list)
- inplace_missings(df, column, method, n_neighbors=2)
- función de máximos, mínimos y percentiles

class read_preprocess(preprocess):
- view_nan_graph(tabla_nan)
- outlier_detection(df, column_list=[]) 
- leer datos de otros formatos
