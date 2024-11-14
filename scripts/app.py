import tkinter as tk
import numpy as np
import tensorflow as tf
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import os
import sys
import logging as log
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tkinter import ttk  # Importar ttk para usar Notebook

# Configurar el nivel de log
log.basicConfig(level=log.INFO)

log.info("Cargando modelo preentrenado ...")
try:
    model = tf.keras.models.load_model("models/modelo_inception.h5")
    log.info("Modelo cargado exitosamente!")
except Exception as e:
    log.error(f"Error al cargar el modelo: {e}")

# Función para cargar y preprocesar la imagen
def preprocess_image(img_path):
    img = Image.open(img_path).resize((299, 299))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Función para realizar la clasificación y mostrar recomendaciones
def classify_image():
    img_path = filedialog.askopenfilename()
    if img_path:
        img = Image.open(img_path).resize((300, 300))  # Redimensionar imagen a 300x300
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

        processed_img = preprocess_image(img_path)
        predictions = model.predict(processed_img)
        class_idx = np.argmax(predictions)
        probability = round(predictions[0][class_idx] * 100, 2)  # Probabilidad en porcentaje

        class_labels = ["Cordana", "Saludable", "Pestaloptiosis", "Sigatoka"]
        result = class_labels[class_idx]
        
        # Recomendaciones basadas en la clasificación
        recommendations = {
            "Cordana": "Recomendación: Realizar tratamiento con fungicida, asegurarse de que las hojas no estén mojadas.",
            "Saludable": "Recomendación: Mantener las condiciones adecuadas de humedad y luz, y realizar un seguimiento regular.",
            "Pestaloptiosis": "Recomendación: Eliminar las hojas infectadas y aplicar fungicida para prevenir la propagación.",
            "Sigatoka": "Recomendación: Usar fungicidas adecuados y evitar riego por aspersión para reducir la propagación."
        }

        # Mostrar el resultado, las recomendaciones
        result_label.config(text=f"Tu imagen se clasificó como: {result} , con una confianza de: {probability}%")
        recommendation_label.config(text=recommendations[result])

# Configuración de la ventana principal
root = tk.Tk()
root.title("🍌 Clasificador de Enfermedades de Hojas de Banano 🍃")
root.geometry("850x700")  # Ajustar el tamaño de la ventana
root.configure(bg="#2F4F4F")

# Crear el notebook (pestañas)
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# Pestaña 1: Menú principal
menu_tab = tk.Frame(notebook, bg="#2F4F4F")
notebook.add(menu_tab, text="Menú")

# Título de la aplicación (en la pestaña de menú)
title_label = tk.Label(
    menu_tab,
    text="🍌 Clasificador de Enfermedades de Hojas de Banano 🍃",
    font=("Helvetica", 20, "bold"),
    fg="white",
    bg="#2F4F4F",
    pady=10
)
title_label.pack()

# Objetivo del modelo y nombres de las clases en la primera pestaña
model_objective_label = tk.Label(
    menu_tab,
    text="Objetivo del modelo: Detectar enfermedades en hojas de banano y proporcionar recomendaciones de tratamiento.",
    font=("Helvetica", 12),
    fg="white",
    bg="#2F4F4F",
    pady=10
)
model_objective_label.pack()

classes_label = tk.Label(
    menu_tab,
    text="Modelo utilizado: InceptionV3 (Transfer Learning) con Imagenet.",
    font=("Helvetica", 12),
    fg="white",
    bg="#2F4F4F",
    pady=10
)
classes_label.pack()


# Crear imágenes representativas para cada clase en la primera pestaña
class_labels = ["Cordana", "Saludable", "Pestaloptiosis", "Sigatoka"]
img_paths = ["static/img/cordana.jpeg", "static/img/healthy.jpeg", "static/img/pestalotiopsis.jpeg", "static/img/sigatoka.jpeg"]  # Coloca las rutas a las imágenes

# Crear un marco horizontal para las imágenes y etiquetas
images_frame = tk.Frame(menu_tab, bg="#2F4F4F")
images_frame.pack(pady=10)

# Estilos mejorados para imágenes y etiquetas
for idx, label in enumerate(class_labels):
    img = Image.open(img_paths[idx]).resize((180, 180))  # Cambié el tamaño a 180x180
    img_tk = ImageTk.PhotoImage(img)
    
    img_label = tk.Label(images_frame, image=img_tk, bg="#2F4F4F", borderwidth=2, relief="solid", padx=10, pady=10)
    img_label.image = img_tk
    img_label.grid(row=0, column=idx, padx=15, pady=10)

    label_text = tk.Label(images_frame, text=label, font=("Helvetica", 12, "bold"), fg="#F0E68C", bg="#2F4F4F")
    label_text.grid(row=1, column=idx, padx=15, pady=5)


# Botón para abrir la pestaña de clasificación
btn_open_classification = tk.Button(
    menu_tab,
    text="Ir a Clasificación",
    font=("Helvetica", 14, "bold"),
    bg="#4682B4",
    fg="white",
    padx=20,
    pady=10,
    borderwidth=0,
    relief="solid",
    command=lambda: notebook.select(classification_tab)
)
btn_open_classification.pack(pady=20)

# Nombres de los creadores
creators_label = tk.Label(
    menu_tab,
    text="Creado por: Samuel Escalante y Emmanuel Quintero",
    font=("Helvetica", 12, "italic"),
    fg="white",
    bg="#2F4F4F",
    pady=10
)
creators_label.pack()

# Pestaña 2: Clasificación
classification_tab = tk.Frame(notebook, bg="#2F4F4F")
notebook.add(classification_tab, text="Clasificación")

# Marco principal para la clasificación
main_frame = tk.Frame(classification_tab, bg="#2F4F4F")
main_frame.pack(pady=20)

# Botón para cargar imagen con efecto hover
def on_hover(event):
    btn_load.config(bg="#5A9BD5")

def on_leave(event):
    btn_load.config(bg="#4682B4")

btn_load = tk.Button(
    main_frame,
    text="Cargar Imagen",
    font=("Helvetica", 14, "bold"),
    bg="#4682B4",
    fg="white",
    padx=20,
    pady=10,
    borderwidth=0,
    relief="solid",
    command=classify_image
)
btn_load.pack(pady=20)
btn_load.bind("<Enter>", on_hover)
btn_load.bind("<Leave>", on_leave)

# Etiqueta para mostrar imagen cargada
img_label = Label(main_frame, bg="#2F4F4F", borderwidth=2, relief="groove")
img_label.pack(pady=10)

# Etiqueta para mostrar el resultado de la clasificación con probabilidad
result_label = tk.Label(
    main_frame,
    text="Esperando resultado:",
    font=("Helvetica", 16, "bold"),
    fg="white",
    bg="#2F4F4F",
    pady=10
)
result_label.pack(pady=10)

# Etiqueta para mostrar las recomendaciones
recommendation_label = tk.Label(
    main_frame,
    text="Esperando para realizar recomendaciones ...",
    font=("Helvetica", 12),
    fg="white",
    bg="#2F4F4F",
    pady=10
)
recommendation_label.pack(pady=10)

# Ejecutar la aplicación
root.mainloop()
