import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/PIA/games_summary.csv")

# Asegurar conversión de la fecha
df['game_date'] = pd.to_datetime(df['game_date'])

# Separar entrenamiento y la prueba
train_df = df[df['game_date'].dt.month < 9]   # enero–agosto
test_df = df[df['game_date'].dt.month == 9]    # septiembre (ultimo mes de la temporada regular)

# Definir variables predictoras (X) y objetivo (y)
features = ['total_balls', 'total_strikes', 'avg_balls', 'avg_strikes',
            'total_pitches', 'avg_release_speed', 'total_atbats']

X_train = train_df[features]
y_train = train_df['total_hits']

X_test = test_df[features]
y_test = test_df['total_hits']

# Establecer el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones para el mes de septiembre
y_pred = model.predict(X_test)

# Metricas del modelo
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("=== RESULTADOS DEL FORECASTING ===")
print(f"R²: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print("")

# Crear la tabla comparativa real vs predicciones
comparison = pd.DataFrame({
    'game_date': test_df['game_date'].dt.strftime("%d/%m/%y"),
    'total_hits_real': y_test.values,
    'total_hits_pred': y_pred
})

output_path = "/Users/ailtonserna/tareas/Mineria/PIA/September_predictions_fullseason.csv"
comparison.to_csv(output_path, index=False)