# Wine Quality MLOps Pipeline

Proyecto de entrenamiento y validación automatizada para la predicción de calidad de vino tinto.

## Estructura
- `train.py`: Entrena una Regresión Lineal y registra métricas/firmas en MLflow.
- `validate.py`: Valida el modelo físico contra un umbral de MSE.
- `data/`: Contiene el dataset fisicoquímico de vinos (UCI Machine Learning).

## Cómo ejecutar
1. Instalar: `make install`
2. Entrenar: `make train`
3. Validar: `make validate`