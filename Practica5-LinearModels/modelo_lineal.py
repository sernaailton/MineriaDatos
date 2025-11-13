import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = "/Users/ailtonserna/tareas/Mineria/Practica5-LinearModels/Graphics"
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Practica2-Statistics/dodgers2025_statistics.csv")

# Variables
X = df[['avg_speed', 'avg_strikes', 'avg_balls', 'avg_strikeouts', 'pitch_name']]
y = df['avg_batting']

# one-hot para el pitch_name
categorical_features = ['pitch_name']
numeric_features = ['avg_speed', 'avg_strikes', 'avg_balls', 'avg_strikeouts']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop=None), categorical_features)
    ]
)

X_encoded = preprocessor.fit_transform(X)

ohe = preprocessor.named_transformers_['cat']
encoded_cols = ohe.get_feature_names_out(categorical_features)
all_columns = numeric_features + list(encoded_cols)

# Entrenar modelo lineal
model = LinearRegression()
model.fit(X_encoded, y)

# Mostrar el valor de R^2
print("R^2 del modelo:", model.score(X_encoded, y))

# Crear DataFrame para la ecuación de regresión lineal
coef_dict = {col: [coef] for col, coef in zip(all_columns, model.coef_)}
coef_dict['Intercepto'] = [model.intercept_]

# Ordenar columnas de coeficientes
df_coef = pd.DataFrame(coef_dict)
cols_order = ['Intercepto'] + all_columns
df_coef = df_coef[cols_order]

# Guardar CSV
df_coef.to_csv(os.path.join(output_dir, "regresion_coeficientes.csv"), index=False, float_format='%.6f')

# Graficar variables numéricas vs avg_batting
for var in numeric_features:
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x=var, y='avg_batting')
    plt.title(f'{var} vs avg_batting')
    plt.xlabel(var)
    plt.ylabel('avg_batting')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{var}_vs_avg_batting.png'))
    plt.close()

