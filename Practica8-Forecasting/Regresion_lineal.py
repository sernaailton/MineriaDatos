import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

output_dir = "/Users/ailtonserna/tareas/Mineria/Practica8-Forecasting"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "september_games_summary_corrected.csv")

df = pd.read_csv(csv_path)
df['game_date'] = pd.to_datetime(df['game_date'])

# Defino las variables que utilizare
features = ['avg_balls', 'avg_strikes', 'total_pitches', 'avg_release_speed', 'total_atbats']
target = 'total_hits'

# Separamos partidos de entrenamiento y prueba
train_df = df.iloc[:22]  # primeros 22 partidos para entrenar el modelo
test_df = df.iloc[22:]   # últimos 3 partidos para hacer la prueba de prediccion

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones del modelo
y_pred = model.predict(X_test)

# Evaluaciones del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# Guardar CSV de predicciones vs reales
pred_df = test_df[['game_date']].copy()
pred_df['total_hits_real'] = y_test.values
pred_df['total_hits_pred'] = y_pred
pred_csv = os.path.join(output_dir, "september_games_predictions.csv")
pred_df.to_csv(pred_csv, index=False)

# Generar y guardar el gráfico comparativo
plt.figure(figsize=(8,6))
plt.plot(pred_df['game_date'], pred_df['total_hits_real'], marker='o', label='Real')
plt.plot(pred_df['game_date'], pred_df['total_hits_pred'], marker='x', label='Predicción')
plt.title("Predicción vs Total Hits - Últimos 3 partidos")
plt.xlabel("Fecha del juego")
plt.ylabel("Total Hits")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pred_vs_real_last3_games.png"))
plt.close()
