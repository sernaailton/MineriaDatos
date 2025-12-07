import pandas as pd

# Cargar el CSV con los resultados reales
df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Segundas/Poisson_Simulation.csv")

# Resumen por fila
resumen_filas = []
for idx, row in df.iterrows():
    resumen = {
        "Player": row["player"],
        "Pitcher Type": row["pitcher_type"],
        "Hit": f"λ_hit={row['lam_hit']:.3f}, p_over_hit={row['p_over_hit']:.3f}, profit={row['profit_hit']} → {'Over' if row['profit_hit']>0 else 'Under' if row['profit_hit']<0 else 'Evitar'}",
        "Walk": f"λ_bb={row['lam_bb']:.3f}, p_over_bb={row['p_over_bb']:.3f}, profit={row['profit_bb']} → {'Over' if row['profit_bb']>0 else 'Under' if row['profit_bb']<0 else 'Evitar'}",
        "RBI": f"λ_rbi={row['lam_rbi']:.3f}, p_over_rbi={row['p_over_rbi']:.3f}, profit={row['profit_rbi']} → {'Over' if row['profit_rbi']>0 else 'Under' if row['profit_rbi']<0 else 'Evitar'}"
    }
    resumen_filas.append(resumen)

df_resumen_filas = pd.DataFrame(resumen_filas)
df_resumen_filas.to_csv("resumen_filas.csv", index=False)
print("✅ Resumen por fila guardado en resumen_filas.csv")

# Resumen por jugador
jugadores = df['player'].unique()
resumen_jugadores = []

for player in jugadores:
    sub_df = df[df['player'] == player]
    # Promedio de profit por estadística y pitcher_type
    hit_over = 'Over' if sub_df['profit_hit'].mean() > 0 else 'Under'
    walk_over = 'Over' if sub_df['profit_bb'].mean() > 0 else 'Under'
    rbi_over = 'Over' if sub_df['profit_rbi'].mean() > 0 else 'Under'

    resumen_jugadores.append({
        "Player": player,
        "Hit": hit_over,
        "Walk": walk_over,
        "RBI": rbi_over
    })

df_resumen_jugadores = pd.DataFrame(resumen_jugadores)
df_resumen_jugadores.to_csv("resumen_jugadores.csv", index=False)
print("✅ Resumen por jugador guardado en resumen_jugadores.csv")

# Propuesta de apuestas para un partido
def propuesta_apuestas(pitcher_type):
    print(f"\n🎯 Propuesta de apuestas vs Pitcher {pitcher_type}")
    for idx, row in df.iterrows():
        if row['pitcher_type'] == pitcher_type:
            hit_action = 'Apostar Over' if row['profit_hit']>0 else 'Apostar Under' if row['profit_hit']<0 else 'Evitar'
            walk_action = 'Apostar Over' if row['profit_bb']>0 else 'Apostar Under' if row['profit_bb']<0 else 'Evitar'
            rbi_action = 'Apostar Over' if row['profit_rbi']>0 else 'Apostar Under' if row['profit_rbi']<0 else 'Evitar'
            print(f"{row['player']}: Hit → {hit_action}, Walk → {walk_action}, RBI → {rbi_action}")

# Ejemplo: partido contra pitcher zurdo
propuesta_apuestas('L')

# Ejemplo: partido contra pitcher derecho
propuesta_apuestas('R')
