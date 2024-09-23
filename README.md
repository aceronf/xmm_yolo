# xmm_yolo

Este repositorio contiene distintos scripts para la adaptación de los datos de imagen procedentes del observatorio **XMM-Newton** (ESA) al formato apto para entrenar las redes neuronales de **YOLOv8** (You Only Look Once). Asimismo, se incluyen herramientas específicas para validar estos modelos una vez han sido entrenados (fine-tuned). 

Los datos en los que se enfoca el proyecto son imágenes en la banda total de energía (producto OIMAGE) y un fichero de regiones para cada una, con información sobre fuentes detectadas automáticamente y, especialmente, áreas problemáticas delimitadas manualmente por revisores. Las funciones que se recogen aquí permiten visualizar y filtrar estas regiones e imágenes para, finalmente, generar los directorios y ficheros necesarios con los que enseñar a las redes neuronales a reconocer defectos en las imágenes automáticamente. Adicionalmente, en el directorio "validación" se aportan funciones con las que cuantificar el rendimiento de los modelos una vez ajustados. Para más información sobre el observatorio XMM-Newton y sus productos, consultar la documentación oficial de la misión: https://www.cosmos.esa.int/web/xmm-newton.

Repositorio oficial de YOLOv8: https://github.com/ultralytics/ultralytics.git.


