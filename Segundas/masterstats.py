import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# ===================== CONFIG =====================
INPUT_CSV = "/Users/ailtonserna/tareas/Mineria/Practica-1/dodgers2025.csv"
OUTPUT_DIR = "/Users/ailtonserna/tareas/Mineria/Segundas/"

# ===================== CUOTAS REALES =====================
odds_dict = {
    "Andy Pages":       {"hit_over":1.48, "hit_under":1.76, "bb_over":2.15, "bb_under":1.68, "rbi_over":2.3, "rbi_under":1.55},
    "Max Muncy":        {"hit_over":1.4,  "hit_under":1.78, "bb_over":1.9,  "bb_under":1.83, "rbi_over":2.1, "rbi_under":1.68},
    "Will Smith":       {"hit_over":1.36, "hit_under":1.8,  "bb_over":1.95, "bb_under":1.8,  "rbi_over":2.1, "rbi_under":1.68},
    "Miguel Rojas":     {"hit_over":1.7,  "hit_under":1.76, "bb_over":2.35, "bb_under":1.55, "rbi_over":2.4, "rbi_under":1.52},
    "Enrique Hernandez":{"hit_over":1.68, "hit_under":1.74, "bb_over":2.1,  "bb_under":1.71, "rbi_over":2.4, "rbi_under":1.52},
    "Dalton Rushing":   {"hit_over":1.84, "hit_under":1.66, "bb_over":2.5,  "bb_under":1.5,  "rbi_over":2.6, "rbi_under":1.43},
    "Hyeseong Kim":     {"hit_over":1.9,  "hit_under":1.7,  "bb_over":2.6,  "bb_under":1.43, "rbi_over":2.3, "rbi_under":1.55},
    "Tommy Edman":      {"hit_over":1.49, "hit_under":1.76, "bb_over":2.1,  "bb_under":1.71, "rbi_over":2.2, "rbi_under":1.62},
    "Michael Conforto": {"hit_over":1.92, "hit_under":1.6,  "bb_over":2.6,  "bb_under":1.43, "rbi_over":2.8, "rbi_under":1.28},
    "Freddie Freeman":  {"hit_over":1.27, "hit_under":1.88, "bb_over":1.76, "bb_under":1.9,  "rbi_over":2.15,"rbi_under":1.74},
    "Teoscar Hernandez":{"hit_over":1.34, "hit_under":1.8,  "bb_over":1.9,  "bb_under":1.83, "rbi_over":2.1, "rbi_under":1.68},
    "Shohei Ohtani":    {"hit_over":1.26, "hit_under":1.89, "bb_over":1.8,  "bb_under":1.95, "rbi_over":2.15,"rbi_under":1.74},
    "Mookie Betts":     {"hit_over":1.26, "hit_under":1.89, "bb_over":1.85, "bb_under":1.96, "rbi_over":2.3, "rbi_under":1.55},
    "Ben Rortvedt":     {"hit_over":1.83, "hit_under":1.66, "bb_over":2.5,  "bb_under":1.5,  "rbi_over":2.5, "rbi_under":1.48},
    "Esteury Ruiz":     {"hit_over":1.95, "hit_under":1.58, "bb_over":2.65, "bb_under":1.38, "rbi_over":2.8, "rbi_under":1.28},
    "Justin Dean":      {"hit_over":1.83, "hit_under":1.66, "bb_over":2.5,  "bb_under":1.5,  "rbi_over":2.7, "rbi_under":1.33},
    "Alex Freeland":    {"hit_over":1.78, "hit_under":1.74, "bb_over":2.4,  "bb_under":1.58, "rbi_over":2.5, "rbi_under":1.48},
    "Alex Call":        {"hit_over":1.53, "hit_under":1.72, "bb_over":2.15, "bb_under":1.68, "rbi_over":2.4, "rbi_under":1.52}
}

# ===================== NOMBRES =====================
player_names = {
    681624: "Andy Pages", 571970: "Max Muncy", 669257: "Will Smith", 500743: "Miguel Rojas",
    571771: "Enrique Hernandez", 687221: "Dalton Rushing", 808975: "Hyeseong Kim",
    669242: "Tommy Edman", 624424: "Michael Conforto", 518692: "Freddie Freeman",
    606192: "Teoscar Hernandez", 660271: "Shohei Ohtani", 605141: "Mookie Betts",
    666163: "Ben Rortvedt", 665923: "Esteury Ruiz", 681909: "Justin Dean",
    690976: "Alex Freeland", 669743: "Alex Call"
}

# ===================== CARGAR CSV =====================
df = pd.read_csv(INPUT_CSV)
df = df[df["events"].notna()].copy()
df["player_name"] = df["batter"].map(player_names)
df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

# ===================== CALCULAR STATS =====================
df["hit"] = df["events"].apply(lambda x: 1 if x in ["single","double","triple","home_run"] else 0)
df["bb"]  = df["events"].apply(lambda x: 1 if x=="walk" else 0)
df["pa"]  = df["events"].apply(lambda x: 0 if x in ["walk","hit_by_pitch"] else 1)
df["rbi"] = df["post_bat_score"] - df["bat_score"]

# ===================== AGRUPACIÓN POR PARTIDO =====================
df_game = df.groupby(["game_date","player_name","p_throws"], as_index=False).agg({
    "hit":"sum","bb":"sum","pa":"sum","rbi":"sum"
})
df_game.to_csv(OUTPUT_DIR+"Stats_Splits.csv", index=False)

# ===================== ESTADÍSTICAS POR JUGADOR =====================
df_stats = df_game.groupby(["player_name","p_throws"]).agg(
    games=("game_date","count"),
    hits_total=("hit","sum"),
    pa_total=("pa","sum"),
    bb_total=("bb","sum"),
    rbi_total=("rbi","sum")
).reset_index()

df_stats["hit_rate_per_pa"] = df_stats["hits_total"]/df_stats["pa_total"].replace(0,np.nan)
df_stats["bb_rate_per_pa"]  = df_stats["bb_total"]/df_stats["pa_total"].replace(0,np.nan)
df_stats["rbi_rate_per_pa"] = df_stats["rbi_total"]/df_stats["pa_total"].replace(0,np.nan)
df_stats["avg_pa_game"]      = df_stats["pa_total"]/df_stats["games"]

# Probabilidades reales de ≥1 HIT/B/ RBI por partido
df_stats["prob_hit_game"] = 1 - (1 - df_stats["hit_rate_per_pa"])**df_stats["avg_pa_game"]
df_stats["prob_bb_game"]  = 1 - (1 - df_stats["bb_rate_per_pa"])**df_stats["avg_pa_game"]
df_stats["prob_rbi_game"] = 1 - (1 - df_stats["rbi_rate_per_pa"])**df_stats["avg_pa_game"]

df_stats.to_csv(OUTPUT_DIR+"Probabilidades.csv", index=False)

# ===================== VALUE BETS =====================
for stat in ["hit","bb","rbi"]:
    df_stats[f"odds_over_{stat}"]  = df_stats["player_name"].map(lambda x: odds_dict[x][f"{stat}_over"])
    df_stats[f"odds_under_{stat}"] = df_stats["player_name"].map(lambda x: odds_dict[x][f"{stat}_under"])
    df_stats[f"p_over_{stat}"]     = 1/df_stats[f"odds_over_{stat}"]
    df_stats[f"p_under_{stat}"]    = 1/df_stats[f"odds_under_{stat}"]
    df_stats[f"value_{stat}_over"] = df_stats[f"prob_{stat}_game"] > df_stats[f"p_over_{stat}"]
    df_stats[f"value_{stat}_under"]= (1-df_stats[f"prob_{stat}_game"]) > df_stats[f"p_under_{stat}"]

df_stats.to_csv(OUTPUT_DIR+"ValueBets.csv", index=False)

# ===================== CLUSTERING =====================
cluster_data = df_stats.groupby("player_name").agg(
    hits=("hits_total","sum"),
    pa=("pa_total","sum"),
    bb=("bb_total","sum"),
    rbis=("rbi_total","sum")
)
cluster_data["contact_rate"]    = cluster_data["hits"]/cluster_data["pa"]
cluster_data["discipline_rate"]= cluster_data["bb"]/(cluster_data["pa"]+cluster_data["bb"])
cluster_data["power_rate"]      = cluster_data["rbis"]/cluster_data["pa"]

X = cluster_data[["contact_rate","discipline_rate","power_rate"]]
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_data["cluster"] = kmeans.fit_predict(X)
cluster_data.to_csv(OUTPUT_DIR+"Clustering.csv")

# ===================== PREDICCIONES POR EVENTO =====================
pred_cols = []
for stat in ["hit","bb","rbi"]:
    y = (df_game[stat]>0).astype(int)  # Evento binario por partido
    X_pred = df_game[["pa"]]          # Usamos PA como predictor simple
    model = LogisticRegression()
    model.fit(X_pred, y)
    df_game[f"pred_prob_{stat}"] = model.predict_proba(X_pred)[:,1]
    pred_cols.append(f"pred_prob_{stat}")

# Guardamos predicciones
df_game.to_csv(OUTPUT_DIR+"Predicciones.csv", index=False)