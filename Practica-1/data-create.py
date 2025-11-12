import warnings #Ignorar warnings por la cantidad tan grande de datos
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
    print(f"Descargando datos desde {inicio} hasta {fin} …")
    data = statcast(start_dt=inicio, end_dt=fin)

    # Logica para verificar que el bateador en el plato sea de los Dodgers
    dodgers_batting = data[
        ((data["inning_topbot"] == "Top") & (data["away_team"] == "LAD")) |
        ((data["inning_topbot"] == "Bot") & (data["home_team"] == "LAD"))
    ]
    all_data.append(dodgers_batting)

dodgers2025 = pd.concat(all_data, ignore_index=True)

# Guardar CSV
output_path = "/Users/ailtonserna/tareas/Mineria/dodgers2025.csv"
dodgers2025.to_csv(output_path, index=False)

