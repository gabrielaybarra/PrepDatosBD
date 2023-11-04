#%%
from preprocess import *

df = pd.read_csv("../matches_1991_2023.csv")

clase = Preprocess(df)
clase.describe_var(['home_team','away_team'], 'cat')
# %%
clase.view_nan_table()
# %%
clase.drop_column(['Referee'])
# %%
clase.inplace_missings('home_goal', 'mode')
# %%
clase2 = Read_Preprocess(Preprocess(df))
# %%
clase2.file_to_dataframe("../matches_1991_2023.csv")
# %%
clase2.outlier_detection(['Glucose','BloodPressure'])
# %%
clase2.view_nan_graph(clase2.view_nan_table())
# %%
