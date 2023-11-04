import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import seaborn as sns
import json
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
import xml.etree.ElementTree as ET
import numpy as np



class Preprocess():

    def describe_var(df, variables, tipo_var):

        """This method will be used to describe one or more columns from a dataframe. 
        The description will be: Count, min, pct 25, mean, median, pct 75, max, std, NaN count and not NaN count

        :param dataframe df: The dataframe from where one or more columns will be described
        :param list variables: Column or list of columns that will be described
        :param str tipo_var: Type of columns that will be described, it manages to options, "cat" or "num"

        :returns: dataframe

        :rtype: dataframe
        """
        try:
            if tipo_var not in ['num', 'cat']:
                raise ValueError("El tipo de variable debe ser 'num' para variables numéricas o 'cat' para variables categóricas.")
            
            if tipo_var == 'num':
                for variable in variables:
                    if df[variable].dtype not in ['int64', 'float64']:
                        raise TypeError(f"La variable '{variable}' no es numérica.")
                
                data = []
                for variable in variables:
                    count = df[variable].count()
                    min_val = df[variable].min()
                    pct25 = np.nanpercentile(df[variable], 25)
                    mean = df[variable].mean()
                    median = df[variable].median()
                    pct75 = np.nanpercentile(df[variable], 75)
                    max_val = df[variable].max()
                    std_dev = df[variable].std()
                    na_count = df[variable].isna().sum()
                    notna_count = df[variable].notna().sum()

                    data.append([
                        count, min_val, pct25, mean, median, pct75, max_val, std_dev, na_count, notna_count
                    ])

                result_df = pd.DataFrame(data, columns=['Count', 'Min', '25th Percentile', 'Mean', 'Median', '75th Percentile', 'Max', 'Std Dev', 'NA Count', 'Not NA Count'], index=variables)
                return result_df.T
            
            elif tipo_var == 'cat':
                for variable in variables:
                    if df[variable].dtype not in ['object', 'category']:
                        raise TypeError(f"La variable '{variable}' no es categórica.")
                
                cat_data = []
                for variable in variables:
                    count = df[variable].count()
                    mode_val = df[variable].mode().iloc[0]
                    mode_freq = df[variable].value_counts().iloc[0]
                    mode_percentage = (mode_freq / count) * 100
                    unique_categories = df[variable].nunique()

                    cat_data.append([
                        count, unique_categories, mode_val, mode_freq, mode_percentage
                    ])

                cat_result_df = pd.DataFrame(cat_data, columns=['Count', 'Unique Categories', 'Mode', 'Mode Frequency', 'Mode Percentage'], index=variables)
                return cat_result_df.T
        
        except (ValueError, TypeError) as e:
            return str(e)
        
       
    def view_nan_table(df):

        """This method is used to generate and view a NaN table. It contains the number of missing values and the percentage of them for each column.

        :param dataframe df: The dataframe from where the missing values is extracted

        :returns: The NaN table

        :rtype: dataframe
        """
    
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Invalid input: 'df' must be a pandas DataFrame.")

            na = df.isna().sum()
            nona = df.notna().sum()
            pct = (na / len(df)) * 100
            total = list(zip(na, nona, pct))
            tabla = pd.DataFrame(total, index=df.columns)
            tabla.columns = ['NaN', 'not_NaN', 'pct']
            tabla['pct'] = round(tabla['pct'].astype(float), 2)

            tabla = tabla.sort_values(by='pct', ascending=False)

            return tabla

        except TypeError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unkown error: {e}")
    

    def drop_column(df, column_list):

        """This method is used to drop one or more columns from a dataframe

        :param dataframe df: The dataframe from where one or more columns is deleted
        :param list column_list: Column or list of columns that is deleted

        :returns: dataframe

        :rtype: dataframe
        """

        try:
            df.drop(column_list, axis=1, inplace=True)
        except KeyError as e:
            print(f"Error: {e} column not found in the DataFrame. Please check the column name.")
        return df
    

    def inplace_missings(df, column, method, n_neighbors = 2):

        """This method inplaces missing values of a given table with the method wanted.

            :param dataframe df: The table where the missing values will be filled
            :param str column: Column of the df to inplace missing values
            :param str method: The method that will be used to inplace the missing values
            :param int n_neighbors: Only used if the method chosen is KNN to inplace missings

            :returns: dataframe with the missing values replaced

            :rtype: dataframe

        """
        try:
            if df[column].isna().sum() == 0:
                raise Exception("There is no missing data in the selected column")
        
            else:
                if method == 'mean':
                    mean = df[column].mean()
                    df[column].fillna(mean, inplace = True)

                if method == 'before':
                    df[column].fillna(method='ffill', inplace = True)

                if method == 'after':
                    df[column].fillna(method='bfill', inplace = True)

                if method == 'mode':
                    mode = df[column].mode()[0]
                    df[column].fillna(mode, inplace = True) 

                if method == 'median':
                    median = df[column].median()
                    df[column].fillna(median, inplace = True)

                if method == 'KNN':
                    knn_imputer = KNNImputer(n_neighbors)
                    imputed_data = knn_imputer.fit_transform(df)
                    df = pd.DataFrame(imputed_data, columns=df.columns)
                
        except KeyError as e:
            print(f"Error: Column not recognized. Please check the data.")
            
        except TypeError as e:
            print(f"Error: Invalid value type detected. Please check the data.")

        except ValueError as e:
            print("Error: Invalid value detected. Please check the data.")
            
        except Exception as e:
            print(f"Error desconocido: {e}")

        return df


## A PARTIR DE AQUI CLASE HEREDADA

class Read_Preprocess(Preprocess):
    def __init__(self):
        super().__init__()

    def file_to_dataframe(path):
        """This method will be used parse files from several extensions to a pandas dataframe

        :param path to file path: Path to the file with a certain extension to be parsed

        :returns: Dataframe

        :rtype: Pandas Dataframe
        """
        extension = path.split('.')[-1].lower()

        if extension == 'json':
            with open(path, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)


        elif extension == 'csv':
            df = pd.read_csv(path)

        elif extension in ('xlsx', 'xls'):
            try:
                workbook = load_workbook(path, read_only=True, data_only=True)
                sheet = workbook.active
                data = sheet.values
                columns = next(data)
                df = pd.DataFrame(data, columns=columns)
            except InvalidFileException:
                df = pd.read_excel(path, engine='xlrd')
        
        elif extension == 'xml':
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                data = []
                for child in root:
                    data.append([child.tag] + [subchild.text for subchild in child])
                columns = data[0]
                data = data[1:]
                df = pd.DataFrame(data, columns=columns)
            except ET.ParseError:
                raise ValueError("El archivo XML no contiene datos en formato tabular")
        
        elif extension == 'h5':
            df = pd.read_hdf(path)
        
        elif extension == 'txt':
            try:
                df = pd.read_csv(path, sep='\t')
            except pd.errors.ParserError:
                raise ValueError("El archivo txt no contiene datos en formato tabular")

        else:
            raise ValueError("Extensión de archivo no compatible")
        
        return df


    def outlier_detection(df, column_list = []):
        """This method will be used to plot and detect outliers from one or more columns

        :param dataframe df: The dataframe from where the outliers will be graphed and detected
        :param list column_list: Column or list of columns from where the outliers will be graphed and detected

        :returns: Graph

        :rtype: Boxplot
        """

        if len(column_list) > 0:
            try:
                plt.figure(figsize = (15,8))
                sns.boxplot(df[column_list], orient='v')
                plt.title('Boxplots of numerical columns', size = 20)
                plt.xlabel('Selected columns', size = 12)
                plt.ylabel('Value', size = 12)
                plt.show()

            except KeyError as e:
                print(f"Error: Invalid value type detected. Please check the data.")
                
            except Exception as e:
                print(f"Error: Unsupported column. Please check the data.")

        else:
            plt.figure(figsize = (15,8))
            sns.boxplot(df, orient='v')
            plt.title('Boxplots of numerical columns', size = 20)
            plt.xlabel('Columns', size = 12)
            plt.ylabel('Value', size = 12)
            plt.show()  

    def view_nan_graph(tabla_nan):

        """This method is used to graph the missing values of a dataframe.

            :param dataframe table_nan: The NaN table obtained from the view_table_nan() method

            :returns: NaN barplot

            :rtype: dataframe
        """

        try:
            # Verifica si tabla_nan es un DataFrame de pandas
            if not isinstance(tabla_nan, pd.DataFrame):
                raise TypeError("Invalid input: 'tabla_nan' must be a pandas DataFrame.")

            # Verifica si hay al menos una columna en el DataFrame
            if len(tabla_nan.columns) == 0:
                raise ValueError("Invalid input: 'tabla_nan' must have at least one column.")

            plt.figure(figsize=(15, 8))
            plt.bar(tabla_nan.index, tabla_nan['pct'])
            plt.title("Pct NaNs", size=20)
            plt.xticks(rotation=90)
            plt.ylabel("Percentage (%)", size=12)
            plt.xlabel("Columns", size=12)
            plt.show()

        except TypeError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unknown error: {e}")

    


