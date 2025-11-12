import pandas as pd
from scipy import stats
from itertools import combinations
import os

df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Practica2-Statistics/dodgers2025_statistics.csv")

# Variables tomadas en cuenta
numeric_columns = ['avg_speed', 'avg_strikes', 'avg_balls', 'avg_batting']

# Grupos por tipo de lanzamiento
pitch_groups = df['pitch_name'].unique()

output_dir = "/Users/ailtonserna/tareas/Mineria/Practica4-StatisticsTest"
os.makedirs(output_dir, exist_ok=True)
results = []

# Realizar la prueba T para los pares de lanzamiento
for col in numeric_columns:
    for g1, g2 in combinations(pitch_groups, 2):
        data1 = df[df['pitch_name'] == g1][col]
        data2 = df[df['pitch_name'] == g2][col]
        
        t_stat, p_val = stats.ttest_ind(data1, data2)
        results.append({
            'variable': col,
            'test_type': 'T-test',
            'group1': g1,
            'group2': g2,
            'statistic': t_stat,
            'p_value': p_val,
            'significant': 'Yes' if p_val < 0.05 else 'No'
        })

results_df = pd.DataFrame(results)
csv_path = os.path.join(output_dir, 'Statistical_T-Test.csv')
results_df.to_csv(csv_path, index=False)