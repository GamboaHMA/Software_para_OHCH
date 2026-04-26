import pandas as pd
from pandas import DataFrame as df

nombres_csv:df = pd.read_csv('nombres_propios/wgnd_source.csv', sep=',')
filtrado:df = nombres_csv[nombres_csv['code'].isin(['ES', 'IT', 'US', 'FR'])]
print(f"tipo de filtrado: {type(filtrado)}")

nombres = (
    filtrado['name']
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .to_list()
)
          
nombres = [nombre.lower() for nombre in nombres]

print('hello')