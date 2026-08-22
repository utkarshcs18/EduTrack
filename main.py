import numpy as np
import pandas as pd

df = pd.read_csv("studentData.csv", encoding="latin1")

print(f'First 5 rows-> \n{df.head()}')
print(f'Last 5 rows-> \n{df.tail()}')

print(df)

row ,column = df.shape

print(f'Rows-> {row}')
print(f'Column-> {column}')

print("Column name : ")
print(df.columns)

print(df.info())

