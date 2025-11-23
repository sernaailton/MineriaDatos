import pandas as pd
import os

output_dir = "/Users/ailtonserna/tareas/Mineria/PIA"
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Practica-1/dodgers2025_clean.csv")

# Convertir fechas de los partidos
df['game_date'] = pd.to_datetime(df['game_date'])

# Definir turnos al bat
df_finished = df[df['events'].notna()].copy()

# Verificar si fue hit
df_finished['hit'] = df_finished['events'].apply(
    lambda x: 1 if x in ['single', 'double', 'triple', 'home_run'] else 0
)

df_finished['balls_adjusted'] = df_finished['balls']
df_finished['strikes_adjusted'] = df_finished['strikes']
df_finished['atbat_adjusted'] = 1  # un turno por fila, excepto walk / hbp

# para strikeout: sumar strike faltante
df_finished.loc[df_finished['events'] == 'strikeout', 'strikes_adjusted'] += 1

# para walk: sumar bola extra y quitar turno al bat
df_finished.loc[df_finished['events'] == 'walk', 'balls_adjusted'] += 1
df_finished.loc[df_finished['events'] == 'walk', 'atbat_adjusted'] -= 1

# para hit by pitch: quitar turno al bat
df_finished.loc[df_finished['events'] == 'hit_by_pitch', 'atbat_adjusted'] -= 1

# Separar las estadisticas de juego por dia
# Turnos totales por juego
turnos_por_juego = df_finished.groupby('game_date')['atbat_adjusted'].sum().rename('total_atbats')

# Sumas de stats
sum_stats = df_finished.groupby('game_date').agg(
    total_hits=('hit', 'sum'),
    total_balls=('balls_adjusted', 'sum'),
    total_strikes=('strikes_adjusted', 'sum'),
    avg_release_speed=('release_speed', 'mean')
)

# Total de lanzamientos por día (TODAS las filas del juego)
total_pitches = df.groupby('game_date').size().rename('total_pitches')

df_grouped = sum_stats.join(turnos_por_juego).join(total_pitches)

# Calcular promedios requeridos
df_grouped['avg_balls'] = df_grouped['total_balls'] / df_grouped['total_atbats']
df_grouped['avg_strikes'] = df_grouped['total_strikes'] / df_grouped['total_atbats']

# Ordenar columnas
df_grouped = df_grouped.reset_index()[[
    'game_date', 'total_hits', 'total_balls', 'total_strikes',
    'avg_balls', 'avg_strikes', 'total_pitches',
    'avg_release_speed', 'total_atbats'
]]

output_csv = os.path.join(output_dir, "games_summary.csv")
df_grouped.to_csv(output_csv, index=False)