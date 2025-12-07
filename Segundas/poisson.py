import pandas as pd
import numpy as np
from scipy.stats import poisson

# ================= CARGAR CSV =====================
df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Segundas/Stats_Splits.csv")
df["game_date"] = pd.to_datetime(df["game_date"])

# ================= CARGAR CUOTAS ==================
odds_data = {
    "Andy Pages": {"hit": (1.48, 1.76), "bb": (2.15,1.68), "rbi": (2.3,1.55)},
    "Max Muncy": {"hit": (1.4,1.78), "bb": (1.9,1.83), "rbi": (2.1,1.68)},
    "Will Smith": {"hit": (1.36,1.8), "bb": (1.95,1.8), "rbi": (2.1,1.68)},
    "Miguel Rojas": {"hit": (1.7,1.76), "bb": (2.35,1.55), "rbi": (2.4,1.52)},
    "Enrique Hernandez": {"hit": (1.68,1.74), "bb": (2.1,1.71), "rbi": (2.4,1.52)},
    "Dalton Rushing": {"hit": (1.84,1.66), "bb": (2.5,1.5), "rbi": (2.6,1.43)},
    "Hyeseong Kim": {"hit": (1.9,1.7), "bb": (2.6,1.43), "rbi": (2.3,1.55)},
    "Tommy Edman": {"hit": (1.49,1.76), "bb": (2.1,1.71), "rbi": (2.2,1.62)},
    "Michael Conforto": {"hit": (1.92,1.6), "bb": (2.6,1.43), "rbi": (2.8,1.28)},
    "Freddie Freeman": {"hit": (1.27,1.88), "bb": (1.76,1.9), "rbi": (2.15,1.74)},
    "Teoscar Hernandez": {"hit": (1.34,1.8), "bb": (1.9,1.83), "rbi": (2.1,1.68)},
    "Shohei Ohtani": {"hit": (1.26,1.89), "bb": (1.8,1.95), "rbi": (2.15,1.74)},
    "Mookie Betts": {"hit": (1.26,1.89), "bb": (1.85,1.96), "rbi": (2.3,1.55)},
    "Ben Rortvedt": {"hit": (1.83,1.66), "bb": (2.5,1.5), "rbi": (2.5,1.48)},
    "Esteury Ruiz": {"hit": (1.95,1.58), "bb": (2.65,1.38), "rbi": (2.8,1.28)},
    "Justin Dean": {"hit": (1.83,1.66), "bb": (2.5,1.5), "rbi": (2.7,1.33)},
    "Alex Freeland": {"hit": (1.78,1.74), "bb": (2.4,1.58), "rbi": (2.5,1.48)},
    "Alex Call": {"hit": (1.53,1.72), "bb": (2.15,1.68), "rbi": (2.4,1.52)}
}

# ================= PREPARAR DATOS =================
df = df[["game_date", "player_name", "p_throws", "hit", "bb", "rbi"]]

# ================= FUNCIÓN PARA POISSON =================
def poisson_prob(lam):
    """Calcula P(over 0.5) y P(under 0.5)"""
    p_over = 1 - poisson.pmf(0, lam)
    p_under = poisson.pmf(0, lam)
    return p_over, p_under

# ================= AGRUPAR POR JUGADOR + PITCHER =================
results = []

for player in df["player_name"].unique():
    df_player = df[df["player_name"] == player].sort_values("game_date")
    for pitcher_type in df_player["p_throws"].unique():
        df_ptype = df_player[df_player["p_throws"] == pitcher_type]
        
        # Excluir últimos 10 partidos
        df_train = df_ptype.iloc[:-10]
        df_test  = df_ptype.iloc[-10:]
        if len(df_train) == 0:
            continue
        
        # Calcular medias λ
        lam_hit = df_train["hit"].mean()
        lam_bb  = df_train["bb"].mean()
        lam_rbi = df_train["rbi"].mean()
        
        # Probabilidades Poisson
        p_over_hit, p_under_hit = poisson_prob(lam_hit)
        p_over_bb, p_under_bb   = poisson_prob(lam_bb)
        p_over_rbi, p_under_rbi = poisson_prob(lam_rbi)
        
        # Cuotas reales
        odds_hit_over, odds_hit_under = odds_data[player]["hit"]
        odds_bb_over, odds_bb_under   = odds_data[player]["bb"]
        odds_rbi_over, odds_rbi_under = odds_data[player]["rbi"]
        
        # Simulación de apuestas
        total_profit = {"hit":0, "bb":0, "rbi":0}
        for idx, row in df_test.iterrows():
            # HIT
            bet_hit = "over" if p_over_hit > p_under_hit else "under"
            actual_hit = "over" if row["hit"] >= 1 else "under"
            if bet_hit == "over":
                total_profit["hit"] += (odds_hit_over - 1)*100 if actual_hit=="over" else -100
            else:
                total_profit["hit"] += (odds_hit_under - 1)*100 if actual_hit=="under" else -100
            # BB
            bet_bb = "over" if p_over_bb > p_under_bb else "under"
            actual_bb = "over" if row["bb"] >= 1 else "under"
            if bet_bb == "over":
                total_profit["bb"] += (odds_bb_over - 1)*100 if actual_bb=="over" else -100
            else:
                total_profit["bb"] += (odds_bb_under - 1)*100 if actual_bb=="under" else -100
            # RBI
            bet_rbi = "over" if p_over_rbi > p_under_rbi else "under"
            actual_rbi = "over" if row["rbi"] >= 1 else "under"
            if bet_rbi == "over":
                total_profit["rbi"] += (odds_rbi_over - 1)*100 if actual_rbi=="over" else -100
            else:
                total_profit["rbi"] += (odds_rbi_under - 1)*100 if actual_rbi=="under" else -100
        
        # Guardar resultados
        results.append({
            "player": player,
            "pitcher_type": pitcher_type,
            "lam_hit": lam_hit,
            "lam_bb": lam_bb,
            "lam_rbi": lam_rbi,
            "p_over_hit": p_over_hit,
            "p_over_bb": p_over_bb,
            "p_over_rbi": p_over_rbi,
            "profit_hit": total_profit["hit"],
            "profit_bb": total_profit["bb"],
            "profit_rbi": total_profit["rbi"]
        })

# ================= GUARDAR RESULTADOS =================
df_results = pd.DataFrame(results)
df_results.to_csv("/Users/ailtonserna/tareas/Mineria/Segundas/Poisson_Simulation.csv", index=False)