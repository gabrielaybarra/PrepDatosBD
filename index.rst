.. Data Preprocessing documentation master file, created by
   sphinx-quickstart on Thu Oct 26 17:59:58 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Data Preprocessing's documentation!
==============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. class:: Preprocess

This class is used to perform basic data processing by means of different specific functions.

   .. method:: view_nan_table(df)

      This method is used to generate and view a NaN table. It contains the number of missing values and the percentage of them for each column.

      :param dataframe df: The dataframe from where the missing values will be extracted

      :returns: The NaN table

      :rtype: dataframe

   .. method:: view_nan_graph(df)

      This method is used to graph the missing values of a dataframe.

      :param dataframe table_nan: The NaN table obtained from the view_table_nan() method

      :returns: NaN barplot

      :rtype: dataframe

   .. method:: drop_column(df, column_list)

      This method will be used to drop one or more columns from a dataframe.

      :param dataframe df: The dataframe from where one or more columns will be deleted.
      :param list column_list: Column or list of columns that will be deleted.

      :returns: dataframe

      :rtype: dataframe
   
   .. method:: inplace_missings(df, column, method, n_neighbors = 2)

      This method inplaces missing values of a given table with the method wanted.

      :param dataframe df: The table where the missing values will be filled
      :param str column: Column of the df to inplace missing values
      :param str method: The method that will be used to inplace the missing values
      :param int n_neighbors: Only used if the method chosen is KNN to inplace missings

      :returns: dataframe with the missing values replaced

      :rtype: dataframe

   .. method:: outlier_detection(df, column_list = [])

      This method will be used to plot and detect outliers from one or more columns

      :param dataframe df: The dataframe from where the outliers will be graphed and detected
      :param list column_list: Column or list of columns from where the outliers will be graphed and detected

      :returns: Graph

      :rtype: Boxplot
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
