# Instala las dependencias del proyecto
install:
	pip install -r requirements.txt

# Entrena el modelo, genera la firma en MLflow y exporta el archivo model.pkl
train:
	python train.py

# Valida el modelo cargando el archivo físico y verificando el umbral de MSE
validate:
	python validate.py