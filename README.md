# PrepDatosBD
## Read, Preprocess and Visualize your data

This library aims to simplify and agilize the process of data preprocessing and cleaning, which is critical in any data analysis or machine learning project. By providing a variety of tools and functions, users can work more efficiently and ensure the quality of the data they are working with.

## Required libraries
```sh
- import pandas as pd
- import matplotlib.pyplot as plt
- from sklearn.impute import KNNImputer
- import seaborn as sns
- import json
- import csv
- from openpyxl import load_workbook
- import xlrd
- from openpyxl.utils.exceptions import InvalidFileException
- import xml.etree.ElementTree as ET
- import numpy as np
```

## Available classes
## Initial class: Preprocess
This class is used to perform basic data processing by means of different specific functions.

*def describe_var(self, variables, tipo_var):*
> This method will be used to describe one or more columns from a dataframe. 
The description will be: Count, min, pct 25, mean, median, pct 75, max, std, NaN count and not NaN count.

*def view_nan_table(self):*
> This method is used to generate and view a NaN table. It contains the number of missing values and the percentage of them for each column.

*def drop_column(self, column_list):*
> This method will be used to drop one or more columns from a dataframe.

*def inplace_missings(self, column, method, n_neighbors=2):*
> This method inplaces missing values of a given table with the method wanted.

## Inherited class: ReadPreprocess
This inherited class is used to perform more advanced data processing by means of different specific functions.

*def file_to_dataframe(self, path):*
> This method will be used parse files from several extensions to a pandas dataframe.

*def outlier_detection(self, df, column_list=[]):*
> This method will be used to plot and detect outliers from one or more columns.

*def view_nan_graph(self, nan_table):*
> This method is used to graph the missing values of a dataframe.

## License

MIT

*Free Software, Hell Yeah!*
