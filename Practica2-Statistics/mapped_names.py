import pandas as pd
df = pd.read_csv("/Users/ailtonserna/tareas/Mineria/Practica-1/dodgers2025_clean.csv")

# Crear diccionario de ID a nombre
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

# Reemplazar columna 'batter' por nombres
df['batter'] = df['batter'].map(batter_nombres)

# Guardar CSV actualizado
output_path = "/Users/ailtonserna/tareas/Mineria/dodgers2025_update.csv"
df.to_csv(output_path, index=False)