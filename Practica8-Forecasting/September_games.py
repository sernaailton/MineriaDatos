import pandas as pd
import os

output_dir = "/Users/ailtonserna/tareas/Mineria/Practica8-Forecasting"
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Practica-1/dodgers2025_clean.csv")

# Filtramos el periodo de tiempo, seleccionando solamente los partidos de Septiembre
df['game_date'] = pd.to_datetime(df['game_date'])
df_sep = df[df['game_date'].dt.month == 9].copy()

# Contamos solamente los registros donde se terminen los turnos al bate
df_finished = df_sep[df_sep['events'].notna()].copy()

# Crear la columna para definir un hit
df_finished['hit'] = df_finished['events'].apply(lambda x: 1 if x in ['single', 'double', 'triple', 'home_run'] else 0)

# Ajustar las bolas y strikes según reglas especiales
# Copiamos las columnas originales
df_finished['balls_adjusted'] = df_finished['balls']
df_finished['strikes_adjusted'] = df_finished['strikes']
df_finished['atbat_adjusted'] = 1  # por defecto ya que cada fila es un turno

# Para el strikeout: agregar 1 strike extra al conteo, ya que el registro no lo cuenta porque se termina el turno antes de contar el strike 3
df_finished.loc[df_finished['events'] == 'strikeout', 'strikes_adjusted'] += 1

# Para el walk: agregar 1 bola extra al conteo, ya que el registro no la cuenta porque se termina el turno antes de contar la bola 4
# Para el walk y hit by pitch: restar un turno al bate, ya que estadisticamente este metodo de embasarse no cuenta como turno al bate, pero el registro si lo esta contando
df_finished.loc[df_finished['events'] == 'walk', 'balls_adjusted'] += 1
df_finished.loc[df_finished['events'].isin(['walk', 'hit_by_pitch']), 'atbat_adjusted'] -= 1

# Calcular las estadísticas para cada partido

# Número de turnos totales por juego
turnos_por_juego = df_finished.groupby('game_date')['atbat_adjusted'].sum().rename('total_atbats')

# Suma de hits, bolas y strikes totales por juego
sum_stats = df_finished.groupby('game_date').agg(
    total_hits=('hit', 'sum'),
    total_balls=('balls_adjusted', 'sum'),
    total_strikes=('strikes_adjusted', 'sum'),
    avg_release_speed=('release_speed', 'mean')
)

# Combinar con número de turnos
df_grouped = sum_stats.join(turnos_por_juego)

# Calcular promedio de bolas y strikes por turno del juego
df_grouped['avg_balls'] = df_grouped['total_balls'] / df_grouped['total_atbats']
df_grouped['avg_strikes'] = df_grouped['total_strikes'] / df_grouped['total_atbats']

# Total de lanzamientos en el juego (incluye todas las filas del juego, es decir cuenta todos los lanzamientos realizados)
total_pitches = df_sep.groupby('game_date').size().rename('total_pitches')
df_grouped = df_grouped.join(total_pitches)

# Ordenar las columnas
df_grouped = df_grouped.reset_index()[[
    'game_date', 'total_hits', 'total_balls', 'total_strikes', 
    'avg_balls', 'avg_strikes', 'total_pitches', 'avg_release_speed', 'total_atbats'
]]

# Guardar CSV
output_csv = os.path.join(output_dir, "September_games_summary_corrected.csv")
df_grouped.to_csv(output_csv, index=False)