import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Practica-1/dodgers2025_clean.csv")

# Se filtran solamente las filas que indiquen que se termino el turno al bate y se crea la columna de estos turnos
df = df[df['events'].notna()].copy()
df['atbat_number'] = df.groupby('batter').cumcount() + 1

# Se crea la variable hit, verificando que la casilla events, determine que el turno terminó en un hit
df['hit'] = df['events'].apply(lambda x: 1 if x in ['single', 'double', 'triple', 'home_run'] else 0)

# Seleccionamos las variables a tomar en cuenta
num_cols = ['release_speed', 'balls', 'strikes']
cat_cols = ['pitch_name']

# Se agrupan por turnos al bate
group_cols = ['batter', 'atbat_number']
df_grouped = df.groupby(group_cols).agg(
    {**{col: 'mean' for col in num_cols}, 'hit': 'max', 'pitch_name': lambda x: x.mode()[0]}
).reset_index()

encoder = OneHotEncoder(sparse_output=False, drop=None)
pitch_encoded = encoder.fit_transform(df_grouped[['pitch_name']])
pitch_df = pd.DataFrame(pitch_encoded, columns=encoder.get_feature_names_out(['pitch_name']))
df_final = pd.concat([df_grouped.drop('pitch_name', axis=1), pitch_df], axis=1)

X = df_final.drop(['batter', 'atbat_number', 'hit'], axis=1)
y = df_final['hit']

# Se rebalancean los datos con SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Escalado de variables numéricas
scaler = StandardScaler()
X_res_scaled = X_res.copy()
X_res_scaled[num_cols] = scaler.fit_transform(X_res[num_cols])

# Busqueda del mejor valor para la K
best_k = 1
best_acc = 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    X_train, X_test, y_train, y_test = train_test_split(X_res_scaled, y_res, test_size=0.2, random_state=42)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    if acc > best_acc:
        best_acc = acc
        best_k = k

# Entrenar KNN con el mejor k obtenido
knn = KNeighborsClassifier(n_neighbors=best_k)
X_train, X_test, y_train, y_test = train_test_split(X_res_scaled, y_res, test_size=0.2, random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Se obtienen los resultados
print(f"Mejor k: {best_k}")
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy del modelo KNN: {acc:.4f}")
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(cm)
report = classification_report(y_test, y_pred)
print("\nReporte de clasificación:")
print(report)


# Guardamos todos los graficos que se realizan
output_dir = "/Users/ailtonserna/tareas/Mineria/Practica6-DataClassification/"
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión")
plt.ylabel("Clase verdadera")
plt.xlabel("Clase predicha")
plt.savefig(output_dir + "matriz_confusion.png")
plt.close()

class_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(6,4))
sns.barplot(x=[0,1], y=class_acc)
plt.title("Accuracy por clase")
plt.xlabel("Clase")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.savefig(output_dir + "accuracy_clase.png")
plt.close()