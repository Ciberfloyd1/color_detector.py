# color_detector.py

red neuronal convolucional (CNN) para detectar y clasificar colores en imágenes. Este ejemplo será una versión simplificada que puede detectar colores básicos como rojo, verde, azul, amarillo, etc.

Primero, necesitaremos instalar algunas dependencias. Puedes instalarlas usando pip:

```plaintext
pip install tensorflow opencv-python numpy matplotlib
```

Ahora, aquí está el script Python que crea y entrena una red neuronal para detectar colores:

3. Ejecuta el script desde la línea de comandos:

```plaintext
python color_detector.py
```


4. La primera vez que ejecutes el script, entrenará un nuevo modelo. Esto puede tardar unos minutos dependiendo de tu hardware.
5. Una vez que el modelo esté entrenado, el script te pedirá que ingreses la ruta a una imagen:

```plaintext
Enter the path to your image (or 'q' to quit):
```


6. Ingresa la ruta completa a tu imagen. Por ejemplo:

```plaintext
Enter the path to your image (or 'q' to quit): C:\Users\YourName\Pictures\red_apple.jpg
```


7. El script analizará la imagen y te dará una predicción del color predominante junto con un nivel de confianza:

```plaintext
The predicted color is red with 98.76% confidence.
```


8. Puedes seguir ingresando rutas de imágenes para analizar más imágenes. Cuando hayas terminado, ingresa 'q' para salir del programa.


Recuerda que este modelo está entrenado para reconocer colores predominantes en imágenes simples. Puede tener dificultades con imágenes complejas o con múltiples colores.

Este script hace lo siguiente:

1. Importa las bibliotecas necesarias.
2. Define una función `generate_color_data()` para crear un conjunto de datos de entrenamiento con imágenes de colores.
3. Genera los datos de entrenamiento.
4. Normaliza los datos y los divide en conjuntos de entrenamiento y prueba.
5. Crea un modelo de red neuronal convolucional.
6. Compila y entrena el modelo.
7. Evalúa el modelo con el conjunto de prueba.
8. Define una función `predict_color()` para predecir el color de una imagen dada.
9. Proporciona un ejemplo de cómo usar la función de predicción.
10. Visualiza el historial de entrenamiento.

11. coloque mejoras las cuales este script mejorado incluye las siguientes características:

1. Más colores: ahora incluye blanco, negro y gris además de los colores anteriores.
2. Generación de datos mejorada: utiliza el espacio de color HSV para generar muestras más realistas.
3. Aumento de datos: añade ruido y variación a las imágenes generadas.
4. Modelo más grande: aumenta el tamaño de las imágenes de entrada a 64x64 píxeles.
5. Guardado y carga del modelo: el modelo entrenado se guarda en un archivo para su uso posterior.
6. Interfaz de usuario mejorada: permite al usuario ingresar rutas de imágenes para predecir colores.
