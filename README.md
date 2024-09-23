# xmm_yolo

Este repositorio contiene distintos ficheros Python para la adaptación de los datos de imagen procedentes del observatorio **XMM-Newton** (ESA) al formato apto para entrenar las redes neuronales de **YOLOv8** (You Only Look Once). Asimismo, se incluyen herramientas específicas para validar estos modelos una vez han sido entrenados (fine-tuned). 

Los datos en los que se enfoca el proyecto son imágenes en la banda total de energía (producto OIMAGE) y un fichero de regiones para cada una, con información sobre fuentes detectadas automáticamente y, especialmente, áreas problemáticas delimitadas manualmente por revisores. Las funciones que se recogen aquí permiten visualizar y filtrar estas regiones e imágenes para, finalmente, generar los directorios y ficheros necesarios con los que enseñar a las redes neuronales a reconocer defectos en las imágenes automáticamente. Adicionalmente, en el directorio "validación" se aportan funciones con las que cuantificar el rendimiento de los modelos una vez ajustados. Para más información sobre el observatorio **XMM-Newton** y sus productos, consultar la documentación oficial de la misión: https://www.cosmos.esa.int/web/xmm-newton.

**YOLOv8** (You Only Look Once) es la última versión de una arquitectura de redes neuronales desarrollada por Ultralytics para realizar, con gran velocidad y precisión, distintas tareas sobre imágenes como clasificación, detección y segmentación. Este proyecto aplica las tareas de *Object Detection* y *OBB* (*Oriented Bounding Boxes Object Detection*).
Repositorio oficial de YOLOv8: https://github.com/ultralytics/ultralytics.git.

## Contenido
### Muestra de imágenes y regiones
En los directorios `oimages_4xmm_image\` y `polyreg_files_4xmm_image\` se incluyen respectivamente 300 imágenes (OIMAGE) de prueba y 300 ficheros de regiones correspondientes a las mismas.

### Procesamiento previo de los datos:
Ficheros dedicados a la preparación de los datos (imágenes y regiones) para su utilización en el entrenamiento de YOLOv8:

- `regiones.py`: Contiene el código básico para procesar ficheros de regiones en formato DS9 y generar una lista de todas las regiones contenidas en ellos, conservando toda su información en forma de atributos (forma, coordenadas, etiquetas, etc.)
- `visualizador.py`: Contiene varias funciones útiles para manejar y visualizar imágenes FITS y archivos de regiones al mismo tiempo. En particular, `ver_todo()` permite visualizar todas las imágenes de un directorio con sus respectivas regiones superpuestas.
- `filtro.py`: Contiene las funciones necesarias para filtrar una serie de observaciones de acuerdo con las regiones que contienen y sus tags. También incluye una función para realizar un filtrado manual.
- `generador_YOLO_detection.py`: Contiene las funciones necesarias para generar los datasets de entrenamiento y validación para la tarea de *Object Detection* en YOLOv8 a partir de los archivos fits y las regiones originales. La función `generar_YOLO()` crea un directorio con la estructura necesaria para llevar a cabo el proceso de entrenamiento.
- `generador_YOLO_OBB.py`: Contiene las funciones necesarias para generar los datasets de entrenamiento y validación para la tarea de *Oriented Bounding Boxes* en YOLOv8 a partir de los archivos fits y las regiones originales. La función `generar_YOLO()` crea un directorio con la estructura necesaria para llevar a cabo el proceso de entrenamiento.

### Validación de modelos entrenados:
Ficheros dedicados a la validación de los modelos resultantes tras el proceso de entrenamiento:

- `predicciones_detection.py`: Contiene el código para predecir sobre imágenes usando un modelo de *Object Detection* de YOLO entrenado previamente. Las predicciones del modelo se comparan con el etiquetado manual ("correcto") a través de tres métricas distintas.
- `predicciones_OBB.py`: Contiene el código para predecir sobre imágenes usando un modelo de *Oriented Bounding Boxes* de YOLO entrenado previamente. Las predicciones del modelo se comparan con el etiquetado manual ("correcto") a través de tres métricas distintas.


## Requisitos
- `Python 3.11`
- `matplotlib`
- `astropy`
- `scikit-learn`
- `PIL`
- `shapely`
- `tifffile`
- `shutil`
- `sys`
- `os`
