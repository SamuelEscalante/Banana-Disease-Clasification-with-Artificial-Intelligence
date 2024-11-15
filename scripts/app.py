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
        # Cargar la imagen y redimensionarla
        img = Image.open(img_path).resize((300, 300))  # Redimensionar imagen a 300x300
        img_tk = ImageTk.PhotoImage(img)  # Convertirla para usar en Tkinter
        img_label.config(image=img_tk)
        img_label.image = img_tk  # Mantener la referencia a la imagen
        
        # Procesar la imagen para la predicción
        processed_img = preprocess_image(img_path)
        predictions = model.predict(processed_img)
        
        # Obtener el índice y probabilidad de la clase con mayor puntaje
        class_idx = np.argmax(predictions)
        probability = round(predictions[0][class_idx] * 100, 2)  # Probabilidad en porcentaje
        
        # Mostrar el resultado, las recomendaciones y las probabilidades de todas las clases
        class_labels = ["Cordana", "Saludable", "Pestaloptiosis", "Sigatoka"]
        result = class_labels[class_idx]
        result_label.config(text=f"Tu imagen se clasificó como: {result} , con una confianza de: {probability}%")
        
        probabilities_text = "Probabilidades:\n"
        for i, label in enumerate(class_labels):
            probabilities_text += f"{label}: {round(predictions[0][i] * 100, 2)}%\n"
            probabilities_label.config(text=probabilities_text)

# Función para regresar al menú principal
def go_to_menu():
    notebook.select(menu_tab)

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
img_refs = {}
for idx, label in enumerate(class_labels):
    img = Image.open(img_paths[idx]).resize((180, 180))  # Cambié el tamaño a 180x180
    img_tk = ImageTk.PhotoImage(img)
    img_refs[label] = img_tk  # Guardar la referencia de la imagen
    
    img_label = tk.Label(images_frame, image=img_tk, bg="#2F4F4F", borderwidth=2, relief="solid", padx=10, pady=10)
    img_label.image = img_tk
    img_label.grid(row=0, column=idx, padx=15, pady=10)

    label_text = tk.Label(images_frame, text=label, font=("Helvetica", 12, "bold"), fg="#F0E68C", bg="#2F4F4F")
    label_text.grid(row=1, column=idx, padx=15, pady=5)

# Crear un marco para los botones
buttons_frame = tk.Frame(menu_tab, bg="#2F4F4F")
buttons_frame.pack(pady=20)

# Botón para abrir la pestaña de clasificación
btn_open_classification = tk.Button(
    buttons_frame,
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
btn_open_classification.grid(row=0, column=0, padx=10)

# Botón para abrir la pestaña de recomendaciones
btn_open_recommendations = tk.Button(
    buttons_frame,
    text="Ir a Recomendaciones",
    font=("Helvetica", 14, "bold"),
    bg="#4682B4",
    fg="white",
    padx=20,
    pady=10,
    borderwidth=0,
    relief="solid",
    command=lambda: notebook.select(recommendations_tab)
)
btn_open_recommendations.grid(row=0, column=1, padx=10)

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

# Etiqueta para mostrar las probabilidades de todas las clases
probabilities_label = tk.Label(
    main_frame,
    text="Probabilidades:",
    font=("Helvetica", 12),
    fg="white",
    bg="#2F4F4F",
    pady=10
)
probabilities_label.pack(pady=10)

# Botón de atrás para regresar al menú principal
back_icon = ImageTk.PhotoImage(Image.open("static/img/barra-de-menus.png").resize((70, 70)))
btn_back_classification = tk.Button(
    classification_tab,
    image=back_icon,
    bg="#2F4F4F",
    borderwidth=0,
    command=go_to_menu
)
btn_back_classification.place(relx=1.0, x=-10, y=10, anchor='ne')  # Colocar el botón en la esquina superior derecha

# Pestaña 3: Recomendaciones
recommendations_tab = tk.Frame(notebook, bg="#2F4F4F")
notebook.add(recommendations_tab, text="Recomendaciones")

btn_back_recommendations = tk.Button(
    recommendations_tab,
    image=back_icon,
    bg="#2F4F4F",
    borderwidth=0,
    command=go_to_menu
)
btn_back_recommendations.place(relx=1.0, x=-10, y=10, anchor='ne')

# Título de la pestaña de recomendaciones
recommendations_title_label = tk.Label(
    recommendations_tab,
    text="Recomendaciones de Tratamiento",
    font=("Helvetica", 20, "bold"),
    fg="white",
    bg="#2F4F4F",
    pady=10
)
recommendations_title_label.pack()

# Descripción de la pestaña de recomendaciones
recommendations_description_label = tk.Label(
    recommendations_tab,
    text="Haga clic en una imagen para ver las recomendaciones de tratamiento.",
    font=("Helvetica", 12),
    fg="white",
    bg="#2F4F4F",
    pady=10
)
recommendations_description_label.pack()

# Diccionario de recomendaciones
recommendations = {
    "Cordana": "Recomendación: Realizar tratamiento con fungicida, asegurarse de que las hojas no estén mojadas.",
    "Saludable": "Recomendación: Mantener las condiciones adecuadas de humedad y luz, y realizar un seguimiento regular.",
    "Pestaloptiosis": "Recomendación: Eliminar las hojas infectadas y aplicar fungicida para prevenir la propagación.",
    "Sigatoka": "Recomendación: Usar fungicidas adecuados y evitar riego por aspersión para reducir la propagación."
}

# Función para mostrar recomendaciones basadas en la selección del usuario
def show_recommendation(label):
    recommendation_text.set(recommendations[label])
    
    # Mostrar la imagen de referencia
    recommendation_img_label.config(image=img_refs[label])
    recommendation_img_label.image = img_refs[label]

# Crear un marco horizontal para las imágenes y etiquetas en la pestaña de recomendaciones
recommendations_images_frame = tk.Frame(recommendations_tab, bg="#2F4F4F")
recommendations_images_frame.pack(pady=10)

# Estilos mejorados para imágenes y etiquetas en la pestaña de recomendaciones
for idx, disease in enumerate(class_labels):
    disease_img_label = tk.Label(recommendations_images_frame, image=img_refs[disease], bg="#2F4F4F", borderwidth=2, relief="solid", padx=10, pady=10)
    disease_img_label.image = img_refs[disease]
    disease_img_label.grid(row=0, column=idx, padx=15, pady=10)
    disease_img_label.bind("<Button-1>", lambda event, dis=disease: show_recommendation(dis))

    disease_label_text = tk.Label(recommendations_images_frame, text=disease, font=("Helvetica", 12, "bold"), fg="#F0E68C", bg="#2F4F4F")
    disease_label_text.grid(row=1, column=idx, padx=15, pady=5)

# Variable para mostrar la recomendación
recommendation_text = tk.StringVar()
recommendation_text.set("Seleccione una enfermedad para ver la recomendación.")

# Etiqueta para mostrar la recomendación
recommendation_display_label = tk.Label(
    recommendations_tab,
    textvariable=recommendation_text,
    font=("Helvetica", 12),
    fg="white",
    bg="#2F4F4F",
    wraplength=600,
    pady=10
)
recommendation_display_label.pack(pady=10)

# Etiqueta para mostrar la imagen de la enfermedad seleccionada
recommendation_img_label = tk.Label(recommendations_tab, bg="#2F4F4F", borderwidth=2, relief="groove")
recommendation_img_label.pack(pady=10)

# Ejecutar la aplicación
root.update() 
root.mainloop()
