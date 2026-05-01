import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import pathlib

def train_model():
    # 1. Configuración de rutas (Compatibilidad Windows/Linux)
    workspace_dir = os.getcwd()
    mlruns_dir = os.path.join(workspace_dir, "mlruns")
    tracking_uri = pathlib.Path(mlruns_dir).as_uri()
    
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Wine_Quality_Regression")

    # 2. Carga de Dataset Externo (Criterio: Dataset Externo)
    data_path = os.path.join(workspace_dir, "data", "winequality-red.csv")
    df = pd.read_csv(data_path, sep=',')
    
    # Separar features y target (quality)
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Entrenamiento
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # 4. Registro en MLflow con Firma y Ejemplo (Criterio: Se excede)
    signature = infer_signature(X_test, preds)
    input_example = X_test.iloc[[0]] # Primera fila como ejemplo

    with mlflow.start_run():
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        print(f"✅ Entrenamiento completado. MSE: {mse:.4f}")

    # 5. Exportación física para el pipeline de CI/CD
    joblib.dump(model, "model.pkl")
    print("✅ Archivo model.pkl generado.")

if __name__ == "__main__":
    train_model()