import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import seaborn as sns


class Preprocess():

    def view_nan_table(df):

        """This method is used to generate and view a NaN table. It contains the number of missing values and the percentage of them for each column.

        :param dataframe df: The dataframe from where the missing values is extracted

        :returns: The NaN table

        :rtype: dataframe
        """
    
        try:
            # Verifica si df es un DataFrame de pandas
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

