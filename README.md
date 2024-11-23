<h1 align="center">🍌 Banana Leaf Disease Classification 🍃</h1>


<h2 align="center">Descripción</h2>

El Clasificador de Enfermedades de Hojas de Banano es una aplicación de aprendizaje automático diseñada para detectar enfermedades en hojas de banano y proporcionar recomendaciones de tratamiento. Utiliza un modelo preentrenado InceptionV3 para la clasificación de imágenes y está construido con TensorFlow y Tkinter para la interfaz gráfica de usuario.

<h2 align="center">Ejemplos de clasificación</h2>

<div align="center">
    <table>
        <tr>
            <td align="center">
                <img src="https://i.postimg.cc/mZJzgWLT/62.jpg" alt="Healthy Leaf" width="200" style="border-radius:10px; border:2px solid #4CAF50;">
                <p><b style="color:#4CAF50;">Healthy</b></p>
            </td>
            <td align="center">
                <img src="https://i.postimg.cc/4nRxbZyt/18.jpg" alt="Sigatoka Disease" width="200" style="border-radius:10px; border:2px solid #4CAF50;">
                <p><b style="color:#4CAF50;">Sigatoka</b></p>
            </td>
            <td align="center">
                <img src="https://i.postimg.cc/m29ccpkC/12.jpg" alt="Cordana Disease" width="200" style="border-radius:10px; border:2px solid #4CAF50;">
                <p><b style="color:#4CAF50;">Cordana</b></p>
            </td>
            <td align="center">
                <img src="https://i.postimg.cc/BZkNDv5y/1.jpg" alt="Pestalotiopsis Disease" width="200" style="border-radius:10px; border:2px solid #4CAF50;">
                <p><b style="color:#4CAF50;">Pestalotiopsis</b></p>
            </td>
        </tr>
    </table>
</div>

<h2 align="center">Sobre los datos</h2>

El dataset de imagenes que se utilizo para realizar este proyecto fue obtenido de:

- [BananaLSD](https://www.kaggle.com/datasets/shifatearman/bananalsd?select=BananaLSD) - Banana Leaf Spot Diseases (BananaLSD) Dataset 🍃

<h2 align="center">Características</h2>

- **Clasificación de Imágenes**: Los usuarios pueden subir imágenes de hojas de banano para clasificarlas en una de cuatro categorías: Cordana, Saludable, Pestalotiopsis y Sigatoka.
- **Probabilidades y Entropía**: La aplicación muestra la probabilidad de clasificación y la entropía de las predicciones.
- **Recomendaciones**: Basado en el resultado de la clasificación, la aplicación proporciona recomendaciones de tratamiento.
- **Interfaz Amigable**: Construida con Tkinter, la aplicación tiene una interfaz intuitiva con pestañas para clasificación y recomendaciones.


<h2 align="center">Requisitos</h2>

- Python 3.x
- TensorFlow
- NumPy
- Tkinter
- PIL (Pillow)
- SciPy
- tqdm

<h2 align="center">Instalación</h2>

1. Clona el repositorio:
   
   ```bash
   git clone https://github.com/SamuelEscalante/Banana-Disease-Clasification-with-Artificial-Intelligence.git
   cd Banana-Disease-Clasification-with-Artificial-Intelligence

2. Instala los paquetes requeridos:

   ```bash
   pip install -r requirements.txt

3. Descarga el modelo preentrenado y colócalo en el directorio models.

4. Prepara el conjunto de datos:
   
  - Organiza tu conjunto de datos en el directorio OriginalSet con subdirectorios para cada clase (cordana, saludable, pestalotiopsis, sigatoka).
  - Ejecuta el script data_splitting.py para dividir el conjunto de datos en conjuntos de entrenamiento, validación y prueba.

<h2 align="center">Uso</h2>

1. Ejecuta la aplicación:

   ```bash
   python scripts/app.py

2. Usa el botón "Cargar Imagen" para subir una imagen de una hoja de banano.

3. La aplicación clasificará la imagen y mostrará el resultado junto con las probabilidades para cada clase.

4. Haz clic en las imágenes en la pestaña "Recomendaciones" para ver las recomendaciones de tratamiento para cada enfermedad.

<h2 align="center">Autores</h2>

- [@SamuelEscalante](https://github.com/SamuelEscalante)
- [@Emmanuel Quintero](https://github.com/EmmanuelQuintero)

---

¡Gracias por visitar nuestro repositorio! Esperamos que encuentre útil este proyecto. Si ha sido útil o simplemente te gustó, ¡considera darle una estrella al repositorio! 🌟

Nos encantaría escuchar sus comentarios, sugerencias o contribuciones.

¡Gracias por tu apoyo! 👋
