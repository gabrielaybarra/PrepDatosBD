import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import seaborn as sns
import json
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
import xml.etree.ElementTree as ET
import numpy as np



class Preprocess:

    def __init__(self, df):
        self.df = df

    def describe_var(self, variables, tipo_var):

        """This method will be used to describe one or more columns from a dataframe. 
        The description will be: Count, min, pct 25, mean, median, pct 75, max, std, NaN count and not NaN count

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
                    if self.df[variable].dtype not in ['int64', 'float64']:
                        raise TypeError(f"La variable '{variable}' no es numérica.")
                
                data = []
                for variable in variables:
                    count = self.df[variable].count()
                    min_val = self.df[variable].min()
                    pct25 = np.nanpercentile(self.df[variable], 25)
                    mean = self.df[variable].mean()
                    median = self.df[variable].median()
                    pct75 = np.nanpercentile(self.df[variable], 75)
                    max_val = self.df[variable].max()
                    std_dev = self.df[variable].std()
                    na_count = self.df[variable].isna().sum()
                    notna_count = self.df[variable].notna().sum()

                    data.append([
                        count, min_val, pct25, mean, median, pct75, max_val, std_dev, na_count, notna_count
                    ])

                result_df = pd.DataFrame(data, columns=['Count', 'Min', '25th Percentile', 'Mean', 'Median', '75th Percentile', 'Max', 'Std Dev', 'NA Count', 'Not NA Count'], index=variables)
                return result_df.T
            
            elif tipo_var == 'cat':
                for variable in variables:
                    if self.df[variable].dtype not in ['object', 'category']:
                        raise TypeError(f"La variable '{variable}' no es categórica.")
                
                cat_data = []
                for variable in variables:
                    count = self.df[variable].count()
                    mode_val = self.df[variable].mode().iloc[0]
                    mode_freq = self.df[variable].value_counts().iloc[0]
                    mode_percentage = (mode_freq / count) * 100
                    unique_categories = self.df[variable].nunique()

                    cat_data.append([
                        count, unique_categories, mode_val, mode_freq, mode_percentage
                    ])

                cat_result_df = pd.DataFrame(cat_data, columns=['Count', 'Unique Categories', 'Mode', 'Mode Frequency', 'Mode Percentage'], index=variables)
                return cat_result_df.T
        
        except (ValueError, TypeError) as e:
            return str(e)
        
       
    def view_nan_table(self):

        """This method is used to generate and view a NaN table. It contains the number of missing values and the percentage of them for each column.

        :returns: The NaN table

        :rtype: dataframe
        """
    
        try:
            if not isinstance(self.df, pd.DataFrame):
                raise TypeError("Invalid input: 'df' must be a pandas DataFrame.")

            na = self.df.isna().sum()
            nona = self.df.notna().sum()
            pct = (na / len(self.df)) * 100
            total = list(zip(na, nona, pct))
            tabla = pd.DataFrame(total, index=self.df.columns)
            tabla.columns = ['NaN', 'not_NaN', 'pct']
            tabla['pct'] = round(tabla['pct'].astype(float), 2)

            tabla = tabla.sort_values(by='pct', ascending=False)

            return tabla

        except TypeError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unkown error: {e}")
    

    def drop_column(self, column_list):

        """This method is used to drop one or more columns from a dataframe

        :param list column_list: Column or list of columns that is deleted

        :returns: dataframe

        :rtype: dataframe
        """

        try:
            self.df.drop(column_list, axis=1, inplace=True)
        except KeyError as e:
            print(f"Error: {e} column not found in the DataFrame. Please check the column name.")
        return self.df
    

    def inplace_missings(self, column, method, n_neighbors = 2):

        """This method inplaces missing values of a given table with the method wanted.

            :param str column: Column of the df to inplace missing values
            :param str method: The method that will be used to inplace the missing values. Possible methods:
                               - 'mean': Inplaces the missings with the mean of the column
                               - 'before': Inplaces the missings with the previous value in the column
                               - 'after': Inplaces the missings with the next value in the column
                               - 'mode': Inplaces the missings with the most common value in the column
                               - 'median': Inplaces the missings with the median of the column
                               - 'KNN': Inplaces the missings using KNN Imputer
            :param int n_neighbors: Only used if the method chosen is 'KNN' to inplace missings

            :returns: dataframe with the missing values replaced

            :rtype: dataframe

        """
        try:
            if self.df[column].isna().sum() == 0:
                raise Exception("There is no missing data in the selected column")
        
            else:
                if method == 'mean':
                    mean = self.df[column].mean()
                    self.df[column].fillna(mean, inplace = True)

                if method == 'before':
                    self.df[column].fillna(method='ffill', inplace = True)

                if method == 'after':
                    self.df[column].fillna(method='bfill', inplace = True)

                if method == 'mode':
                    mode = self.df[column].mode()[0]
                    self.df[column].fillna(mode, inplace = True) 

                if method == 'median':
                    median = self.df[column].median()
                    self.df[column].fillna(median, inplace = True)

                if method == 'KNN':
                    knn_imputer = KNNImputer(n_neighbors)
                    imputed_data = knn_imputer.fit_transform(self.df)
                    self.df = pd.DataFrame(imputed_data, columns=self.df.columns)
                
        except KeyError as e:
            print(f"Error: Column not recognized. Please check the data.")
            
        except TypeError as e:
            print(f"Error: Invalid value type detected. Please check the data.")

        except ValueError as e:
            print("Error: Invalid value detected. Please check the data.")
            
        except Exception as e:
            print(f"Error desconocido: {e}")

        return self.df


class Read_Preprocess(Preprocess):
    def __init__(self, df):
        super().__init__(df)

    def file_to_dataframe(self, path):
        """This method will be used parse files from several extensions to a pandas dataframe

        :param path to file path: Path to the file with a certain extension to be parsed

        :returns: Dataframe

        :rtype: Pandas Dataframe
        """
        extension = path.split('.')[-1].lower()

        if extension == 'json':
            with open(path, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)


        elif extension == 'csv':
            self.df = pd.read_csv(path)

        elif extension in ('xlsx', 'xls'):
            try:
                workbook = load_workbook(path, read_only=True, data_only=True)
                sheet = workbook.active
                data = sheet.values
                columns = next(data)
                self.df = pd.DataFrame(data, columns=columns)
            except InvalidFileException:
                raise Exception("Invalid File Exception. Please check the file.")
        
        elif extension == 'xml':
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                data = []
                for child in root:
                    data.append([child.tag] + [subchild.text for subchild in child])
                columns = data[0]
                data = data[1:]
                self.df = pd.DataFrame(data, columns=columns)
            except ET.ParseError:
                raise ValueError("XML file does not contain data in tabular format.")
        
        elif extension == 'h5':
            self.df = pd.read_hdf(path)
        
        elif extension == 'txt':
            try:
                self.df = pd.read_csv(path, sep='\t')
            except pd.errors.ParserError:
                raise ValueError("txt file does not contain data in tabular format.")

        else:
            raise ValueError("File extension not compatible.")
        
        return self.df


    def outlier_detection(self, column_list = []):
        """This method will be used to plot and detect outliers from one or more columns

        :param list column_list: Column or list of columns from where the outliers will be graphed and detected

        :returns: Graph

        :rtype: Boxplot
        """

        if len(column_list) > 0:
            try:
                plt.figure(figsize = (15,8))
                sns.boxplot(self.df[column_list], orient='v')
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
            sns.boxplot(self.df, orient='v')
            plt.title('Boxplots of numerical columns', size = 20)
            plt.xlabel('Columns', size = 12)
            plt.ylabel('Value', size = 12)
            plt.show()  

    def view_nan_graph(self, nan_table):

        """This method is used to graph the missing values of a dataframe.

            :param dataframe table_nan: The NaN table obtained from the view_table_nan() method

            :returns: NaN barplot

            :rtype: dataframe
        """

        try:
            if not isinstance(nan_table, pd.DataFrame):
                raise TypeError("Invalid input: 'nan_table' must be a pandas DataFrame.")

            if len(nan_table.columns) == 0:
                raise ValueError("Invalid input: 'nan_table' must have at least one column.")

            plt.figure(figsize=(15, 8))
            plt.bar(nan_table.index, nan_table['pct'])
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

    


