import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

output_dir = "/Users/ailtonserna/tareas/Mineria/Practica7-DataClustering"
df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Practica-1/dodgers2025_clean.csv")

# Filtrar lanzamientos que sean validos
df = df[df['release_speed'].notna() & df['balls'].notna() & df['strikes'].notna() & df['pitch_name'].notna()]

# Seleccionamos las variables a tomar en cuenta
num_cols = ['release_speed', 'balls', 'strikes']
cat_cols = ['pitch_name']

encoder = OneHotEncoder(sparse_output=False)
pitch_encoded = encoder.fit_transform(df[cat_cols])
pitch_df = pd.DataFrame(pitch_encoded, columns=encoder.get_feature_names_out(cat_cols))
df_model = pd.concat([df[num_cols].reset_index(drop=True), pitch_df.reset_index(drop=True)], axis=1)

# Escalamos las variables
scaler = StandardScaler()
df_model[num_cols] = scaler.fit_transform(df_model[num_cols])

# Busqueda del mejor valor para la K
inertia = []
K_range = range(1, 16)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df_model)
    inertia.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.title("Método del codo")
plt.xlabel("Número de clusters k")
plt.ylabel("Inertia")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "elbow_method.png"))
plt.close()

# Utilizamos silhouette para seleccionar k
sil_scores = []
for k in range(2, 16):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(df_model)
    sil = silhouette_score(df_model, labels)
    sil_scores.append(sil)

best_k = np.arange(2, 16)[np.argmax(sil_scores)]
print(f"Mejor k seleccionado automáticamente por silhouette: {best_k}")

# Entrenar KMeans con el mejor k obtenido
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df_model)

# Guardo los resultados
df.to_csv(os.path.join(output_dir, "dodgers2025_clusters.csv"), index=False)

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='release_speed', y='strikes', hue='cluster', palette='tab10')
plt.title("Clusters de lanzamientos: release_speed vs strikes")
plt.savefig(os.path.join(output_dir, "release_speed_vs_strikes_clusters.png"))
plt.close()

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='release_speed', y='balls', hue='cluster', palette='tab10')
plt.title("Clusters de lanzamientos: release_speed vs balls")
plt.savefig(os.path.join(output_dir, "release_speed_vs_balls_clusters.png"))
plt.close()

# Informe de clusters
report = []
for i in range(best_k):
    cluster_data = df[df['cluster'] == i]
    report.append({
        'cluster': i,
        'count': len(cluster_data),
        'mean_release_speed': cluster_data['release_speed'].mean(),
        'mean_balls': cluster_data['balls'].mean(),
        'mean_strikes': cluster_data['strikes'].mean()
    })

report_df = pd.DataFrame(report)
report_df.to_csv(os.path.join(output_dir, "cluster_summary.csv"), index=False)