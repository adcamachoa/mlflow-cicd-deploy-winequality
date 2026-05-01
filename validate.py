import joblib
import pandas as pd
import os
import sys
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

THRESHOLD = 0.8 # Umbral exigente para calidad de vino

def validate():
    # Cargar datos para validación
    data_path = os.path.join(os.getcwd(), "data", "winequality-red.csv")
    df = pd.read_csv(data_path, sep=',')
    X = df.drop('quality', axis=1)
    y = df['quality']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Intentar cargar el modelo físico
    if not os.path.exists("model.pkl"):
        print("❌ Error: No se encontró model.pkl")
        sys.exit(1)
        
    model = joblib.load("model.pkl")
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    
    print(f"🔍 MSE Resultante: {mse:.4f}")
    
    if mse < THRESHOLD:
        print("✅ Validación exitosa: El modelo cumple la calidad.")
        sys.exit(0)
    else:
        print(f"❌ Validación fallida: MSE {mse} por encima del umbral.")
        sys.exit(1)

if __name__ == "__main__":
    validate()