import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Practica2-Statistics/dodgers2025_statistics.csv")

# Seleccionar variables numéricas relevantes
num_vars = ['avg_speed', 'avg_strikes', 'avg_balls', 'avg_strikeouts', 'avg_batting']

# Calcular la matriz de correlación
corr_matrix = df[num_vars].corr()

# Crear un heatmap de la correlación
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlación entre variables numéricas")
plt.tight_layout()
# Guardar el gráfico en PNG
plt.savefig("/Users/ailtonserna/tareas/Mineria/Practica5-LinearModels/correlacion_variables.png")
plt.close()

print("Heatmap de correlación guardado correctamente.")
