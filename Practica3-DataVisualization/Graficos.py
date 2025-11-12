import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = "/Users/ailtonserna/tareas/Mineria/Practica3/graphs"
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Practica2-Statistics/dodgers2025_statistics.csv")

# Histogramas
numeric_columns = ['avg_speed', 'avg_batting', 'avg_balls', 'avg_strikes']
for col in numeric_columns:
    plt.figure(figsize=(7,5))
    plt.hist(df[col], bins=15, edgecolor='black')
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/hist_{col}.png")
    plt.close()

# Boxplot comparativo
plt.figure(figsize=(8,6))
df.boxplot(column=numeric_columns)
plt.title("Distribución de variables clave (Boxplot)")
plt.savefig(f"{output_dir}/boxplot_comparativo.png")
plt.close()

# Pie chart de tipos de lanzamiento
pitch_counts = df['pitch_name'].value_counts()
plt.figure(figsize=(7,7))
plt.pie(pitch_counts, labels=pitch_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribución porcentual de tipos de lanzamiento")
plt.savefig(f"{output_dir}/pie_pitch_types.png")
plt.close()

# Scatter plot velocidad vs promedio bateo
plt.figure(figsize=(7,5))
plt.scatter(df['avg_speed'], df['avg_batting'], alpha=0.7)
plt.title("Relación entre velocidad promedio y promedio de bateo")
plt.xlabel("Velocidad promedio (avg_speed)")
plt.ylabel("Promedio de bateo (avg_batting)")
plt.grid(True)
plt.savefig(f"{output_dir}/scatter_speed_vs_batting.png")
plt.close()

# Line plot velocidad promedio por tipo de lanzamiento
avg_speed_by_pitch = df.groupby('pitch_name')['avg_speed'].mean()
plt.figure(figsize=(8,5))
avg_speed_by_pitch.plot(kind='line', marker='o')
plt.title("Velocidad promedio por tipo de lanzamiento")
plt.xlabel("Tipo de lanzamiento (pitch_name)")
plt.ylabel("Velocidad promedio (avg_speed)")
plt.grid(True)
plt.savefig(f"{output_dir}/line_avg_speed_by_pitch.png")
plt.close()

# Gráfico de barras por conteo de lanzamientos por tipo
plt.figure(figsize=(8,5))
pitch_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Cantidad de lanzamientos por tipo")
plt.xlabel("Tipo de lanzamiento")
plt.ylabel("Frecuencia")
plt.grid(axis='y', alpha=0.3)
plt.savefig(f"{output_dir}/bar_pitch_counts.png")
plt.close()

# Histogramas por tipo de lanzamiento (stacked)
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='avg_speed', hue='pitch_name', multiple='stack')
plt.title("Distribución de velocidad por tipo de lanzamiento")
plt.xlabel("Velocidad promedio")
plt.ylabel("Frecuencia")
plt.savefig(f"{output_dir}/hist_speed_by_pitch.png")
plt.close()

# Scatter plot por tipo de lanzamiento
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x='avg_speed', y='avg_batting', hue='pitch_name', palette='Set2')
plt.title("Velocidad vs promedio de bateo por tipo de lanzamiento")
plt.xlabel("Velocidad promedio")
plt.ylabel("Promedio de bateo")
plt.grid(True)
plt.savefig(f"{output_dir}/scatter_speed_vs_batting_by_pitch.png")
plt.close()

# Heatmap de correlación entre variables numéricas
plt.figure(figsize=(7,5))
corr = df[numeric_columns].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Mapa de correlación entre variables")
plt.savefig(f"{output_dir}/heatmap_correlacion.png")
plt.close()

# Boxplot por tipo de lanzamiento
plt.figure(figsize=(8,5))
sns.boxplot(x='pitch_name', y='avg_batting', data=df)
plt.title("Distribución de promedio de bateo por tipo de lanzamiento")
plt.xlabel("Tipo de lanzamiento")
plt.ylabel("Promedio de bateo")
plt.savefig(f"{output_dir}/boxplot_batting_by_pitch.png")
plt.close()

# Gráfico de líneas múltiples para avg_speed y avg_strikes por tipo de lanzamiento
plt.figure(figsize=(8,5))
df.groupby('pitch_name')[['avg_speed','avg_strikes']].mean().plot(marker='o')
plt.title("Velocidad y strikes promedio por tipo de lanzamiento")
plt.xlabel("Tipo de lanzamiento")
plt.ylabel("Promedio")
plt.grid(True)
plt.savefig(f"{output_dir}/line_speed_strikes_by_pitch.png")
plt.close()