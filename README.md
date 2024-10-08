# xmm_yolo

Este repositorio contiene distintos ficheros Python para la adaptación de los datos de imagen procedentes del observatorio **XMM-Newton** (ESA) al formato apto para entrenar las redes neuronales de **YOLOv8** (You Only Look Once). Asimismo, se incluyen herramientas específicas para validar estos modelos una vez han sido entrenados (fine-tuned). 

Los datos en los que se enfoca el proyecto son imágenes en la banda total de energía (producto OIMAGE) y un fichero de regiones para cada una, con información sobre fuentes detectadas automáticamente y, especialmente, áreas problemáticas delimitadas manualmente por revisores. Las funciones que se recogen aquí permiten visualizar y filtrar estas regiones e imágenes para, finalmente, generar los directorios y ficheros necesarios con los que enseñar a las redes neuronales a reconocer defectos en las imágenes automáticamente. Adicionalmente, en el directorio `validación/` se aportan funciones con las que cuantificar el rendimiento de los modelos una vez ajustados. Para más información sobre el observatorio **XMM-Newton** y sus productos, consultar la documentación oficial de la misión: https://www.cosmos.esa.int/web/xmm-newton.

**YOLOv8** (You Only Look Once) es la última versión de una arquitectura de redes neuronales desarrollada para realizar, con gran velocidad y precisión, distintas tareas sobre imágenes como clasificación, detección y segmentación. Este proyecto aplica las tareas de *Object Detection* y *OBB* (*Oriented Bounding Boxes Object Detection*).
Repositorio oficial de YOLOv8: https://github.com/ultralytics/ultralytics.git.

## Contenido
Dentro de cada fichero Python se aporta documentación detallada sobre cada función específica y sus utilidades.

### Muestra de imágenes y regiones:
En los directorios `oimages_4xmm_image/` y `polyreg_files_4xmm_image/` se incluyen respectivamente 300 imágenes (OIMAGE) de prueba y 300 ficheros de regiones correspondientes a las mismas.

### Procesamiento previo de los datos:
Ficheros dedicados a la preparación de los datos (imágenes y regiones) para su utilización en el entrenamiento de YOLOv8:

- `regiones.py`: Contiene el código básico para procesar ficheros de regiones en formato DS9 y generar una lista de todas las regiones contenidas en ellos, conservando toda su información en forma de atributos (forma, coordenadas, etiquetas, etc.)
- `visualizador.py`: Contiene varias funciones útiles para manejar y visualizar imágenes FITS y archivos de regiones al mismo tiempo. En particular, `ver_todo()` permite visualizar todas las imágenes de un directorio con sus respectivas regiones superpuestas.
- `filtro.py`: Contiene las funciones necesarias para filtrar una serie de observaciones de acuerdo con las regiones que contienen y sus tags. También incluye una función para realizar un filtrado manual.
- `generador_YOLO_detection.py`: Contiene las funciones necesarias para generar los datasets de entrenamiento y validación para la tarea de *Object Detection* a partir de los archivos fits y las regiones originales. La función `generar_YOLO()` crea un directorio con la estructura necesaria para llevar a cabo el proceso de entrenamiento en YOLOv8.
- `generador_YOLO_OBB.py`: Contiene las funciones necesarias para generar los datasets de entrenamiento y validación para la tarea de *Oriented Bounding Boxes* a partir de los archivos fits y las regiones originales. La función `generar_YOLO()` crea un directorio con la estructura necesaria para llevar a cabo el proceso de entrenamiento en YOLOv8.

### Validación de modelos entrenados:
Ficheros dedicados a la validación de los modelos resultantes tras el proceso de entrenamiento:

- `predicciones_detection.py`: Contiene el código para predecir sobre imágenes usando un modelo de *Object Detection* de YOLO entrenado previamente. Las predicciones del modelo se comparan con el etiquetado manual ("correcto") mediante tres métricas distintas.
- `predicciones_OBB.py`: Contiene el código para predecir sobre imágenes usando un modelo de *Oriented Bounding Boxes* de YOLO entrenado previamente. Las predicciones del modelo se comparan con el etiquetado manual ("correcto") mediante tres métricas distintas.

### Ejemplos de ejecución:
El directorio `ejemplos/muestra_1/` muestra un ejemplo completo de cómo es posible utilizar el código de este repositorio para adaptar los datos "crudos" de `oimages_4xmm_image/` y `polyreg_files_4xmm_image/` para el entrenamiento de YOLOv8.
El directorio `validación/ejemplo_fold_2/` contiene un ejemplo de validación de un modelo ya entrenado sobre un conjunto de imágenes.

El proceso por el que se han obtenido estos resultados está descrito abajo paso a paso.

## Requisitos
- `Python 3.11`
- `numpy`
- `matplotlib`
- `astropy`
- `scikit-learn`
- `PIL`
- `shapely`
- `tifffile`
- `shutil`
- `sys`
- `os`
- `ultralytics`
- `skimage`

## Ejemplo de uso
### 1. Filtrado de datos
Los directorios `oimages_4xmm_image/` y `polyreg_files_4xmm_image/` contienen los datos correspondientes a 300 imágenes OIMAGE. Es posible filtrarlos para guardar únicamente las imágenes con, al menos, una región poligonal de *tag* "manual". Esto se consigue ejecutando la función `save_polymanreg()`, en `filtro.py`. 

El resultado es una nueva carpeta llamada `muestra_1\`, con las 5 imágenes que cumplen la condición anterior y el fichero de regiones de cada una de ellas.

### 2. Visualización de las imágenes con sus regiones
Para ver las imágenes de `muestra_1/` con sus regiones superpuestas, ejecutamos `ver_todo("muestra_1")`, en el fichero `visualizador.py`. Si además queremos guardar las figuras resultantes en una subcarpeta, ejecutamos `ver_todo("muestra_1", guardar=True)`. Las imágenes se representan en escala de grises siguiendo un código de colores para las regiones: cian (detecciones automáticas de fuentes), blanco (detecciones espurias marcadas automáticamente por el SAS), amarillo (regiones etiquetadas manualmente como "single"), rojo (regiones etiquetadas manualmente como "manual") y verde (etiquetadas manualmente como "source").

![Resultados_visualizador](ejemplos/muestra_1/visualizador/P0008820101EPX000OIMAGE8000.png)

### 3. Creación de los datos de entrenamiento
Para afinar una red neuronal con los datos contenidos en `muestra_1/`, hay que generar una estructura de ficheros específica (véase la documentación de YOLOv8). Esto se puede lograr con la función `generar_YOLO("muestra_1", class_dif=False, draw_circles=True, splits=3)`, del fichero `generar_YOLO_detection.py`. `class_dif=False` indica que las regiones **no** se clasificarán manualmente en función del tipo de defecto; `draw_circles=True` edita las imágenes para que aparezcan sobre ellas las detecciones automáticas sobreimpresas, lo que ha demostrado ser de gran ayuda en el aprendizaje de los modelos; `splits=3` crea 3 iteraciones distintas (folds) de los conjuntos de entrenamiento y validación, dejando en cada caso 2/3 de las imágenes para entrenamiento y 1/3 para validación. 

El resultado es una carpeta llamada `YOLO` con una subcarpeta por cada fold de la validación cruzada, cada una de las cuales se puede utilizar para entrenar a YOLOv8. Las carpetas `images/` contienen las imágenes en formato TIFF, mientras que las carpetas `labels/` contienen un fichero de texto asociado a cada imagen, donde cada línea se corresponde con las coordenadas del rectángulo que engloba una región. No todas las regiones presentes originalmente se incluyen aquí para el entrenamiento: la función `reg_2_YOLO()` las filtra para dejar únicamente las etiquetadas como "manual" o "single" (esto es fácilmente modificable en el código, por ejemplo, si se quiere dejar sólo las regiones etiquetadas con "manual").

En cada fold debe aparecer también un archivo .yaml con las rutas a las imágenes y etiquetas en cada caso. De momento este archivo .yaml no se genera automáticamente, pero se puede encontrar la plantilla en `ejemplos/muestra_1/`.

```
YOLO/
├── fold_1/
│   ├── images/
│   │   ├── train/
│   │   └── validation/
│   ├── labels/
│   │   ├── train/
│   │   └── validation/
│   └── data.yaml
├── fold_2/
│   ├── images/
│   │   ├── train/
│   │   └── validation/
│   ├── labels/
│   │   ├── train/
│   │   └── validation/
│   └── data.yaml
└── fold_3/
    ├── images/
    │   ├── train/
    │   └── validation/
    ├── labels/
    │   ├── train/
    │   └── validation/
    └── data.yaml

```

En este caso, generamos los datos para la tarea sencilla de *Object Detection*. Para la tarea de *Oriented Bounding Boxes* es quivalente, simplemente ejecutando la función `generar_YOLO()` contenida en `generar_YOLO_OBB.py`.

### 4. Entrenamiento (ajuste fino) de un modelo
El proceso de entrenamiento una vez preparado el conjunto de datos es sencillo y se puede llevar a cabo con un comando a través de la terminal. Por ejemplo, una vez situados en la carpeta `muestra_1/YOLO/fold_1/`
```
yolo detect train data=data.yaml model=yolov8n.pt epochs=200 imgsz=648
```

### 5. Validación del modelo
Cuando el entrenamiento ha concluido, YOLO crea en la carpeta del conjunto de datos un directorio llamado `runs/detect/train/` y guarda allí el modelo resultante y distintos datos de validación generados automáticamente. En `validación/ejemplo_fold_2/` se muestra el directorio generado anteriormente, `fold_2/`, después del entrenamiento:

```
├── validación/ejemplo_fold_2/
    ├── images/
    │   ├── train/
    │   └── validation/
    ├── labels/
    │   ├── train/
    │   └── validation/
    ├── data.yaml
    |
    └── runs/
        └── detect/
            └── train/
```
La validación consiste en usar el modelo obtenido para predecir sobre las imágenes de `images/validation/`. Esto se consigue ejecutando la función `pred_todo("ejemplo_fold_2", train="train", save_images=True, classdif=False)`, contenida en `predicciones_detection.py`. `save_images=True` permite guardar en `runs/detect/train/predicciones_val/` dos figuras por cada imagen del conjunto de validación: una con la predicción del modelo (verde) y otra con las anotaciones originales (rojo):

![Resultados_visualizador](validación/ejemplo_fold_2/runs/detect/train/predicciones_val/P0004210201EPX000OIMAGE8000.tif_orig.png)
![Resultados_visualizador](validación/ejemplo_fold_2/runs/detect/train/predicciones_val/P0004210201EPX000OIMAGE8000.tif_pred.png)
