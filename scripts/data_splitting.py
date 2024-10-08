import os
import numpy as np
from glob import glob
from tqdm import tqdm
import logging as log

# Configuración de logging
log.basicConfig(level=log.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Definir nombres de clases
class_names = ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka']

# Definir directorios de entrenamiento, validación y prueba
train_dir = "data/train"
valid_dir = "data/valid"
test_dir  = "data/test"

for directory in [train_dir, valid_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Crear subdirectorios para cada clase en los conjuntos de datos
for name in class_names:
    for directory in [train_dir, valid_dir, test_dir]:
        class_path = os.path.join(directory, name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)

# Recopilar todas las rutas de imágenes para cada clase
all_class_paths = {name: glob(f"AugmentedSet/{name}/*") for name in class_names}

# Definir proporciones para entrenamiento, validación y prueba
train_ratio = 0.70
valid_ratio = 0.10
test_ratio  = 0.20

# Inicializar contadores globales
total_train = 0
total_valid = 0
total_test = 0

# Dividir imágenes para cada clase
for class_name, paths in all_class_paths.items():
    total_images = len(paths)
    
    # Barajar las imágenes
    np.random.shuffle(paths)
    
    # Calcular los tamaños de cada conjunto
    train_size = int(total_images * train_ratio)
    valid_size = int(total_images * valid_ratio)
    test_size  = total_images - train_size - valid_size  # Lo que queda es para test
    
    # Crear listas de imágenes para cada conjunto
    train_images = [(path, os.path.join(train_dir, class_name, os.path.basename(path))) for path in paths[:train_size]]
    valid_images = [(path, os.path.join(valid_dir, class_name, os.path.basename(path))) for path in paths[train_size:train_size + valid_size]]
    test_images  = [(path, os.path.join(test_dir, class_name, os.path.basename(path)))  for path in paths[train_size + valid_size:]]
    
    # Actualizar contadores globales
    total_train += len(train_images)
    total_valid += len(valid_images)
    total_test  += len(test_images)
    
    # Mover las imágenes a sus nuevos directorios
    for images, data_type in [(train_images, "Training"), (valid_images, "Validation"), (test_images, "Testing")]:
        for old_path, new_path in tqdm(images, desc=f"{data_type} Data - {class_name}"):
            os.rename(old_path, new_path)

# Mostrar el total de imágenes por cada conjunto
log.info(f"Total de imágenes en entrenamiento: {total_train}")
log.info(f"Total de imágenes en validación: {total_valid}")
log.info(f"Total de imágenes en prueba: {total_test}")
log.info(f"Total de imágenes: {total_train + total_valid + total_test}")

# Log de confirmación final
log.info("¡Todos los datos han sido movidos correctamente!")
