import warnings  # Ignorar warnings por la cantidad tan grande de datos
warnings.filterwarnings("ignore", category=FutureWarning)

from pybaseball import statcast
import pandas as pd

# Defino periodos para no tener problemas en la descarga de datos
periodos = [
    ("2025-03-17", "2025-05-31"),
    ("2025-06-01", "2025-08-01"),
    ("2025-08-02", "2025-09-29")
]

all_data = []

for inicio, fin in periodos:
    data = statcast(start_dt=inicio, end_dt=fin)

    # Logica para verificar que el bateador en el plato sea de los Dodgers
    dodgers_batting = data[
        ((data["inning_topbot"] == "Top") & (data["away_team"] == "LAD")) |
        ((data["inning_topbot"] == "Bot") & (data["home_team"] == "LAD"))
    ]

    # Seleccionar columnas en el orden indicado
    columnas = [
        'game_date', 'player_name', 'p_throws', 'batter', 'stand',
        'balls', 'strikes', 'pitch_name', 'launch_speed', 'description',
        'events', 'release_speed', 'inning'
    ]
    columnas_existentes = [col for col in columnas if col in dodgers_batting.columns]
    dodgers_batting = dodgers_batting[columnas_existentes]

    all_data.append(dodgers_batting)

dodgers2025 = pd.concat(all_data, ignore_index=True)

# Aqui organizo los datos para tenerlos de manera cronologica
dodgers2025["game_date"] = pd.to_datetime(dodgers2025["game_date"], errors="coerce")

# Identificamos el fin del turno con la columna events
dodgers2025["nuevo_turno"] = dodgers2025["events"].notna().astype(int)

# Crear número incremental de turno dentro de cada juego e inning
dodgers2025["turno_numero"] = (
    dodgers2025.groupby(["game_date", "inning"])["nuevo_turno"]
    .cumsum()
    .add(1)
)

# Ordenar respetando el orden: juego → inning → turno → conteo bolas/strikes
dodgers2025 = dodgers2025.sort_values(
    by=["game_date", "inning", "turno_numero", "balls", "strikes"],
    ascending=[True, True, True, True, True]
)
dodgers2025 = dodgers2025.drop(columns=["nuevo_turno"], errors="ignore")

# Guardar el CSV organizado
output_path = "/Users/ailtonserna/tareas/Mineria/Practica-1/dodgers2025_clean.csv"
dodgers2025.to_csv(output_path, index=False)