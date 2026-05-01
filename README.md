# 🍷 Wine Quality Prediction - MLOps Pipeline

Este repositorio contiene un flujo completo de MLOps para predecir la calidad del vino tinto, integrando automatización de CI/CD y gestión de experimentos.

## 🚀 Características del Proyecto
- **Dataset Externo:** Procesamiento de datos físicos (`winequality-red.csv`).
- **CI/CD:** Pipeline automatizado en GitHub Actions para entrenamiento y validación.
- **MLflow Tracking:** Registro detallado de métricas, parámetros y firmas de modelos.
- **Calidad Asegurada:** Validación automática de métricas de error (MSE) previa a la generación de artefactos.

## 🛠️ Stack Tecnológico
- **Lenguaje:** Python 3.11
- **ML Framework:** Scikit-Learn (Regresión Lineal)
- **Tracking:** MLflow
- **Automatización:** GitHub Actions & Make
- **Análisis de Datos:** Pandas & NumPy

## 📊 Ejecución Local
Para replicar los resultados:
1. Instalar dependencias: `make install`
2. Entrenar el modelo: `make train`
3. Validar calidad: `make validate`
4. Ver UI de MLflow: `mlflow ui`

## 📈 Resultados
El modelo actual presenta un **MSE de 0.39**, superando satisfactoriamente el umbral de calidad definido de 0.8.