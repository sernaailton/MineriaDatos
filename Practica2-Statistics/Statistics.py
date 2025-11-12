import pandas as pd

# Cargar dataset actualizado
df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/dodgers2025_update.csv")

# Revisar columnas
print("Columnas disponibles en el CSV:", df.columns)

# Definir eventos que terminan un turno
turn_end_events = [
    'single', 'double', 'triple', 'home_run',
    'strikeout', 'groundout', 'flyout', 'forceout', 'field_out',
    'walk', 'hit_by_pitch', 'intent_walk', 'sac_fly', 'sac_bunt'
]

# Crear columna para marcar fin de turno
df['turn_end'] = df['events'].isin(turn_end_events)

# Crear ID de turno único por bateador (usando 'batter', el ID)
df['turn_id'] = df.groupby('batter')['turn_end'].cumsum()

# Estadísticas por turno
turn_stats = df.groupby(['batter', 'turn_id', 'pitch_name']).agg(
    hits_turn=('events', lambda x: sum(e in ['single','double','triple','home_run'] for e in x)),
    strikeout_turn=('events', lambda x: sum(e == 'strikeout' for e in x)),
    balls_turn=('balls', 'sum'),
    strikes_turn=('strikes', 'sum'),
    avg_speed_turn=('release_speed', 'mean')
).reset_index()

# Estadísticas por bateador y tipo de lanzamiento
pitch_stats = turn_stats.groupby(['batter', 'pitch_name']).agg(
    num_turns=('turn_id', 'nunique'),
    total_hits=('hits_turn', 'sum'),
    avg_batting=('hits_turn', lambda x: x.sum() / len(x)),
    total_strikeouts=('strikeout_turn', 'sum'),
    avg_strikeouts=('strikeout_turn', lambda x: x.sum() / len(x)),
    avg_balls=('balls_turn', 'mean'),
    avg_strikes=('strikes_turn', 'mean'),
    avg_speed=('avg_speed_turn', 'mean'),
    min_speed=('avg_speed_turn', 'min'),
    max_speed=('avg_speed_turn', 'max'),
    std_speed=('avg_speed_turn', 'std')
).reset_index()

# Guardar CSV final
output_path = "/Users/ailtonserna/tareas/Mineria/dodgers2025_statistics.csv"
pitch_stats.to_csv(output_path, index=False)

print(f"\n✅ CSV final guardado en: {output_path}")
print(f"📊 Total de bateadores por pitch: {len(pitch_stats)}")
print(pitch_stats.head())
