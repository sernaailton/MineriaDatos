import pandas as pd
from scipy import stats
import os

df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Practica2-Statistics/dodgers2025_statistics.csv")

# Variables tomadas en cuenta
numeric_columns = ['avg_speed', 'avg_strikes', 'avg_balls', 'avg_batting']

# Grupos por tipo de lanzamiento
pitch_groups = df['pitch_name'].unique()

output_dir = "/Users/ailtonserna/tareas/Mineria/Practica4-StatisticsTest"
os.makedirs(output_dir, exist_ok=True)
results = []

for col in numeric_columns:
    groups = [df[df['pitch_name'] == pitch][col] for pitch in pitch_groups]
    
    # TABLA ANOVA
    f_stat, p_anova = stats.f_oneway(*groups)
    anova_significant = 'Yes' if p_anova < 0.05 else 'No'
    
    # Kruskal-Wallis
    h_stat, p_kruskal = stats.kruskal(*groups)
    kruskal_significant = 'Yes' if p_kruskal < 0.05 else 'No'
    
    results.append({
        'variable': col,
        'anova_F': f_stat,
        'anova_pvalue': p_anova,
        'anova_significant': 'Yes' if p_anova < 0.05 else 'No',
        'kruskal_H': h_stat,
        'kruskal_pvalue': p_kruskal,
        'kruskal_significant': 'Yes' if p_kruskal < 0.05 else 'No'
    })

results_df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, 'Statistical_Test_Results.csv')
results_df.to_csv(csv_path, index=False)