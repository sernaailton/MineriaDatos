import pandas as pd
import os
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter

output_dir = "/Users/ailtonserna/tareas/Mineria/Practica9-TextAnalysis"
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Practica-1/dodgers2025_clean.csv")

# Se realiza el mapeo de los id de los bateadores para poder leer el nombre, ya que este archivo aun no esta mapeado en la practica 1, que es de donde se lee
batter_nombres = {
    681624: "Andy Pages",
    571970: "Max Muncy",
    669257: "Will Smith",
    500743: "Miguel Rojas",
    571771: "Enrique Hernandez",
    687221: "Dalton Rushing",
    808975: "Hyeseong Kim",
    669242: "Tommy Edman",
    624424: "Michael Conforto",
    518692: "Freddie Freeman",
    606192: "Teoscar Hernandez",
    660271: "Shohei Ohtani",
    605141: "Mookie Betts",
    666163: "Ben Rortvedt",
    665923: "Esteury Ruiz",
    681909: "Justin Dean",
    690976: "Alex Freeland",
    669743: "Alex Call"
}
df['batter'] = df['batter'].map(batter_nombres)

# Se ignora la columa events cuando esta vacia, osea que el lanzamiento fue bola o strike, no termino un turno al bate
df_text = df[df['events'].notna()].copy()

# Combinar las columnas de texto
columns_to_use = ['player_name', 'batter', 'pitch_name', 'description', 'events']
all_words = []
for col in columns_to_use:
    # Tomar valores no nulos
    words = df_text[col].dropna().astype(str).tolist()
    all_words.extend(words)

# Contar las frecuencias de todas las palabras
word_freq = Counter(all_words)

# Generar la wordcloud con las frecuencias
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=1200, height=600,
                      background_color='white',
                      stopwords=stopwords,
                      max_words=500,
                      collocations=False).generate_from_frequencies(word_freq)

# Guardar la imagen de la wordcloud
output_image = os.path.join(output_dir, "dodgers_wordcloud.png")
wordcloud.to_file(output_image)