# -*- coding: utf-8 -*-
"""
Contiene las funciones necesarias para generar los datasets de entrenamiento
y validación para OBB (Oriented Bounding Boxes, marcos orientables) en YOLO a
partir de los archivos fits y las regiones originales.

Clases de defectos:
    0: Straylights
    1: Out-of-time events
    2: Extended emission
    3: CCD Bands
    
@author: Alejandro Cerón Fernández
@date: 03/06/2024
"""

import regiones as rg
from astropy.io import fits
from astropy.visualization import MinMaxInterval, AsymmetricPercentileInterval
from astropy.visualization import SqrtStretch
import matplotlib.pyplot as plt
import os
import sys
import shutil
import tifffile
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.affinity import rotate
from sklearn.model_selection import KFold

def poly_2_bbox(reg, class_num:int)->tuple:
    """
    Función que recibe una región poligonal como parámetro y devuelve una
    tupla con la bounding box orientada en formato YOLO. La bbox es el rectángulo
    más pequeño que abarca toda la región.
    
    Parameters
    ----------
    reg : Region
        Región poligonal con un número arbitrario de vértices.
        
    class_ num : int
        Número entero que indica la clase a la que pertenece el objeto.

    Returns
    -------
    tuple :
        tupla con los parámetros de la bbox en formato YOLO:
        (class centro_x centro_y ancho alto)

    """
    # Se comprueba que la región es poligonal:
    if reg.get_forma() != "polygon":
        sys.exit()
    
    # Puntos que componen el polígono:
    puntos = np.array(list(zip(reg.get_coordenadas()[0::2], reg.get_coordenadas()[1::2])))
    # Creamos un polígono con shapely a partir de estos puntos:
    hull_polygon = Polygon(puntos)
    
    # Lo que haremos ahora será girar el polígono hasta que la bounding box no 
    # orientada tenga la mínima superficie. El ángulo que minimice el área será
    # el ángulo de la bounding box orientada para el polígono original:
    # Inicializamos la variable de área a minimizar:
    min_area = float('inf')
    
    best_bounds = None
    best_angle = None
    
    # Rotamos el polígono de 0 a 180 grados y calculamos el área de la bbox
    # en cada caso, quedándonos con el ángulo que la minimiza:
    for angle in np.arange(0, 180, 1):
        # Rotamos el polígono manteniendo fijo el origen de coordenadas de la imagen:
        rotated_polygon = rotate(hull_polygon, angle, origin=(0,0), use_radians=False)
        # Límites del polígono rotado que definen su bbox: (minx, miny, maxx, maxy)
        bounds = rotated_polygon.bounds
        # Ancho y alto
        ancho = bounds[2] - bounds[0]
        alto = bounds[3] - bounds[1]
        area = ancho * alto
        if area < min_area:
            min_area = area
            best_bounds = bounds
            best_angle = angle
    
    # La bbox del polígono rotado de forma óptima es:
    minx, miny, maxx, maxy = best_bounds
    # Deshacemos el giro de esta bounding box, creando el rectángulo que será
    # la bbox orientada que queremos:
    rect = plt.Rectangle((minx, miny), maxx-minx, maxy-miny, angle=360-best_angle, 
                         rotation_point=(0,0))
    vertices = rect.get_corners() # Vértices del rectángulo
    # Vértices normalizados aptos para OBB:
    vertices_normalizados = tuple([float(coord/648) for vertex in vertices for coord in vertex])
    
    return (class_num,)+vertices_normalizados

    
def box_2_bbox(reg, class_num:int, size_filter:bool)->tuple:
    """
    Función que recibe una región box como parámetro y devuelve una
    tupla con la bbox en formato YOLO. La bbox es el rectángulo más pequeño
    que abarca toda la región.
    
    Parameters
    ----------
    reg : Region
        Región rectangular.
    
    class_ num : int
        Número entero que indica la clase a la que pertenece el objeto.
        
    size_filter : bool
        Indica si se quiere aplicar un filtro para no tener en cuenta los
        rectángulos más pequeños o no.

    Returns
    -------
    tuple :
        tupla con los parámetros de la bbox en formato YOLO:
        (classx1y1x2y2x3y3x4y4)

    """
    # Se comprueba que la región es de tipo box:
    if reg.get_forma() != "box":
        sys.exit()
    
    # Creamos el rectángulo:
    posx, posy, ancho, alto, angle = reg.get_coordenadas() # Coordenadas de la región
    rect = plt.Rectangle((posx - ancho/2, posy - alto/2), ancho, alto, 
                         angle=angle, rotation_point="center", fill=False) # Rectángulo pyplot 
    vertices = rect.get_corners() # Vértices del rectángulo
    
    # Coordenadas x e y de cada vértice del polígono:
    x_coord = [float(vert[0]) for vert in vertices]
    y_coord = [float(vert[1]) for vert in vertices]

    # Dimensiones de la bbox (no orientada) en coordenadas relativas:
    ancho = (max(x_coord) - min(x_coord))/648
    alto = (max(y_coord) - min(y_coord))/648
    
    # Vértices normalizados aptos para OBB:
    vertices_normalizados = tuple([float(coord/648) for vertex in vertices for coord in vertex])
    
    if size_filter: # 0.13 es buen umbral
        if (ancho>0 or alto>0):
            return (class_num,)+vertices_normalizados 
        else:
            return None
    else:
       return (class_num,)+vertices_normalizados 
    
def circle_2_bbox(reg, class_num:int, size_filter:bool)->tuple:
    """
    Función que recibe una región circular como parámetro y devuelve una
    tupla con la bbox en formato YOLO para OBB. La bbox es el rectángulo más pequeño
    que abarca toda la región.
    
    Parameters
    ----------
    reg : Region
        Región circular.
    
    class_ num : int
        Número entero que indica la clase a la que pertenece el objeto.
        
    size_filter : bool
        Indica si se quiere aplicar un filtro para no tener en cuenta los
        círculos más pequeños o no.

    Returns
    -------
    tuple :
        tupla con los parámetros de la bbox en formato YOLO:
        (classx1y1x2y2x3y3x4y4)

    """
    # Se comprueba que la región es de tipo circle:
    if reg.get_forma() != "circle":
        sys.exit()
        
    centro_x, centro_y, radio = reg.get_coordenadas()
    rect = plt.Rectangle((centro_x - radio, centro_y - radio), 2*radio, 2*radio, 
                         angle=0, rotation_point="center", fill=False) # Rectángulo pyplot 
    vertices = rect.get_corners() # Vértices del rectángulo
    
    # Coordenadas x e y de cada vértice del polígono:
    x_coord = [float(vert[0]) for vert in vertices]
    y_coord = [float(vert[1]) for vert in vertices]

    # Dimensiones de la bbox (no orientada) en coordenadas relativas:
    ancho = (max(x_coord) - min(x_coord))/648
    alto = (max(y_coord) - min(y_coord))/648
    
    # Vértices normalizados aptos para OBB:
    vertices_normalizados = tuple([float(coord/648) for vertex in vertices for coord in vertex])
    
    if size_filter: # 0.13 es buen umbral
        if (ancho>0 or alto>0):
            return (class_num,)+vertices_normalizados 
        else:
            return None
    else:
       return (class_num,)+vertices_normalizados 
        
def ellipse_2_bbox(reg, class_num:int, size_filter:bool)->tuple:
    """
    Función que recibe una región ellipse como parámetro y devuelve una
    tupla con la bbox en formato YOLO para OBB. La bbox es el rectángulo más pequeño
    que abarca toda la región, orientado en este caso, pues las elipses tienen
    un ángulo de giro en formato ds9.
    
    Parameters
    ----------
    reg : Region
        Región elíptica.
    
    class_ num : int
        Número entero que indica la clase a la que pertenece el objeto.
        
    size_filter : bool
        Indica si se quiere aplicar un filtro para no tener en cuenta los
        círculos más pequeños o no.

    Returns
    -------
    tuple :
        tupla con los parámetros de la bbox en formato YOLO:
        (classx1y1x2y2x3y3x4y4)

    """
    # Se comprueba que la región es de tipo ellipse:
    if reg.get_forma() != "ellipse":
        sys.exit()
        
    centro_x, centro_y, r1, r2, angle = reg.get_coordenadas()
    rect = plt.Rectangle((centro_x - r1, centro_y - r2), 2*r1, 2*r2, 
                         angle=angle, rotation_point="center", fill=False) # Rectángulo pyplot 
    vertices = rect.get_corners() # Vértices del rectángulo
    
    # Coordenadas x e y de cada vértice del polígono:
    x_coord = [float(vert[0]) for vert in vertices]
    y_coord = [float(vert[1]) for vert in vertices]

    # Dimensiones de la bbox (no orientada) en coordenadas relativas:
    ancho = (max(x_coord) - min(x_coord))/648
    alto = (max(y_coord) - min(y_coord))/648
    
    # Vértices normalizados aptos para OBB:
    vertices_normalizados = tuple([float(coord/648) for vertex in vertices for coord in vertex])
    
    if size_filter: # 0.13 es buen umbral
        if (ancho>0 or alto>0):
            return (class_num,)+vertices_normalizados 
        else:
            return None
    else:
       return (class_num,)+vertices_normalizados 
    
def reg_2_YOLO(image_path:str, class_diff:bool):
    """
    Función que recibe como parámetro la dirección de una imagen fits, lo lee 
    y guarda en una carpeta llamada "YOLO" la imagen procesada junto con un archivo .txt con las
    anotaciones en formato YOLO.
    
    Parameters
    ----------
    image_path : str
        Dirección a la imagen FITS.
        
    class_diff : bool
        Indica si se quiere clasificar los objetos por tipos (True) o 
        adjudicar a todas las regiones la clase 0 (False).

    Returns
    -------
    None.

    """
    # Se abre la imagen fits:
    with fits.open(image_path) as lista_hdu:
        
        image = lista_hdu[0].data
        # Normalización MinMax:
        interval_minmax = MinMaxInterval()
        minmax = interval_minmax(image)
        # Combinación óptima de estiramiento y normalización por percentiles:
        comb = SqrtStretch() + AsymmetricPercentileInterval(0, 99.9)
        comb_im = comb(minmax)
        # Conversión de la imagen a 16-bit
        scaled_image = (comb_im * (2**16 - 1)).astype(np.uint16)
        # Exportamos en FITS modificado:
        nombre_fit = os.path.basename(image_path) # Nombre del archivo fits
        origen = os.path.dirname(image_path) # Directorio donde se encuentra el fits

        destino = os.path.join(origen, "YOLO", os.path.splitext(nombre_fit)[0]+".tif")  
        anotaciones_path = os.path.join(origen, "YOLO", nombre_fit[:27]+".txt")
                                    
        # Se guarda la imagen en formato TIFF    
        tifffile.imwrite(destino, scaled_image)  
            
    # Se lee el archivo de regiones correspondiente:
    nombre_reg = nombre_fit[:11]+"_poly.reg" # Nombre del archivo de regiones
    regiones = rg.reg_lista(os.path.join(origen, nombre_reg)) # Lista de regiones
    # Filtramos esta lista de regiones:
    regiones_filt = [x for x in regiones 
                     if (x.get_tag()=="{man}") and
                     (x.get_forma()=="polygon" or x.get_forma()=="box" or x.get_forma()=="circle" or x.get_forma()=="ellipse")]
    
    # Si class_dif=True, se comienza por representar la imagen
    if class_diff:
        for region in regiones_filt:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(comb_im, cmap='gray')
            # Se pintan todas las regiones
            for r in regiones_filt:
                r.plot_region(ax, destacar=False)
            # Se destaca la región actual
            region.plot_region(ax, destacar=True)
            plt.show()
            
            while True:
                class_num = input("Class number: ")
                if class_num in ["","0","1","2","3","4"]:
                    break
                else:
                    print("Input inválido \n")
            if class_num=="":
                continue
            
            if region.get_forma() == "polygon":
                # Tupla con los parámetros de la región en formato YOLO:
                params = poly_2_bbox(region,int(class_num))
            elif region.get_forma() == "box":
                # Tupla con los parámetros de la región en formato YOLO:
                params = box_2_bbox(region,int(class_num),False)
            elif region.get_forma() == "circle":
                # Tupla con los parámetros de la región en formato YOLO:
                params = circle_2_bbox(region,int(class_num),False)
            elif region.get_forma() == "ellipse":
                # Tupla con los parámetros de la región en formato YOLO:
                params = ellipse_2_bbox(region,int(class_num),False)
                
            else:   
                params = None
            if params is not None:
                with open(anotaciones_path, "a", encoding='utf-8') as file:  
                    file.write(f"{params[0]} {params[1]} {params[2]} "\
                               f"{params[3]} {params[4]} {params[5]} "\
                               f"{params[6]} {params[7]} {params[8]} \n")
    else:
        for region in regiones_filt:   
            if region.get_forma() == "polygon":
                # Tupla con los parámetros de la región en formato YOLO:
                params = poly_2_bbox(region,0)
                            
            elif region.get_forma() == "box":
                # Tupla con los parámetros de la región en formato YOLO:
                params = box_2_bbox(region,0,True)
            
            elif region.get_forma() == "circle":
                # Tupla con los parámetros de la región en formato YOLO:
                params = circle_2_bbox(region,0,True)
            
            elif region.get_forma() == "ellipse":
                # Tupla con los parámetros de la región en formato YOLO:
                params = ellipse_2_bbox(region,0,True)
                
            else:   
                params = None
            
            if params is not None:
                with open(anotaciones_path, "a", encoding='utf-8') as file:  
                    file.write(f"{params[0]} {params[1]} {params[2]} "\
                               f"{params[3]} {params[4]} {params[5]} "\
                               f"{params[6]} {params[7]} {params[8]} \n")
                        
                        
def reg_2_YOLO_circles(image_path:str, class_diff:bool):
    """
    Función que recibe como parámetro la dirección de una imagen fits, lo lee 
    y guarda en una carpeta llamada "YOLO" la imagen procesada junto con un archivo .txt con las
    anotaciones en formato YOLO. En función de si el modo es "train" ò "val", se guardarán 
    los datos en las carpetas que corresponda. En este caso, la imagen guardada
    contiene círculos cian correspondientes a las detecciones automáticas del SAS, 
    sin tag.
    
    Parameters
    ----------
    image_path : str
        Dirección a la imagen FITS.
        
    class_diff : bool
        Indica si se quiere clasificar los objetos por tipos (True) o 
        adjudicar a todas las regiones la clase 0 (False).

    Returns
    -------
    None.

    """
    # Se abre la imagen fits:
    with fits.open(image_path) as lista_hdu:
        
        image = lista_hdu[0].data
        # Normalización MinMax:
        interval_minmax = MinMaxInterval()
        minmax = interval_minmax(image)
        # Combinación óptima de estiramiento y normalización por percentiles:
        comb = SqrtStretch() + AsymmetricPercentileInterval(0, 99.9)
        comb_im = comb(minmax)
        # Conversión de la imagen a 16-bit
        scaled_image = (comb_im * (2**16 - 1)).astype(np.uint16)
        
        # Conversión a RGB
        image_rgb = np.stack((scaled_image,) * 3, axis=-1)
        image_rgb_uint8 = (image_rgb / 256).astype(np.uint8)
        # Creamos un objeto pillow:
        image_pil = Image.fromarray(image_rgb_uint8)
        draw = ImageDraw.Draw(image_pil)
        
        # Se lee el archivo de regiones correspondiente para dibujar las regiones
        # sin tag, circulares o elípticas:
        nombre_fit = os.path.basename(image_path) # Nombre del archivo fits
        origen = os.path.dirname(image_path) # Directorio donde se encuentra el fits
        nombre_reg = nombre_fit[:11]+"_poly.reg" # Nombre del archivo de regiones
        regiones = rg.reg_lista(os.path.join(origen, nombre_reg)) # Lista de regiones
        # Cogemos las regiones sin tag circulares o elípticas:
        detections = [x for x in regiones if x.get_tag()=="{none}" and
                   (x.get_forma()=="circle" or x.get_forma()=="ellipse")]
        # Dibujamos cada una de esas regiones sobre la imagen:
        for reg in detections:
            if reg.get_forma()=="circle":
                x,y,radio = reg.get_coordenadas()
                left_up_point = (x - radio, y - radio)
                right_down_point = (x + radio, y + radio)
                
            if reg.get_forma()=="ellipse":
                x,y,r1,r2,angle = reg.get_coordenadas()
                left_up_point = (x - r1, y - r2)
                right_down_point = (x + r1, y + r2)
                
            draw.ellipse([left_up_point, right_down_point], outline="red", width=1)    
        
        # Convertimos de nuevo a un NumPy array:
        modified_image = np.array(image_pil)
        # Exportamos el FITS modificado:
        destino = os.path.join(origen, "YOLO", os.path.splitext(nombre_fit)[0]+".tif")  
        anotaciones_path = os.path.join(origen, "YOLO", nombre_fit[:27]+".txt")    
        # Se guarda la imagen en formato TIFF:
        tifffile.imwrite(destino, modified_image)
            
    
    # Filtramos esta lista de regiones:
    regiones_filt = [x for x in regiones 
                     if (x.get_tag()=="{man}" or x.get_tag()=="{sin}") and
                     (x.get_forma()=="polygon" or x.get_forma()=="box" or x.get_forma()=="circle" or x.get_forma()=="ellipse")]
    
    # Si class_dif=True, se comienza por representar la imagen
    if class_diff:
        for region in regiones_filt:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(comb_im, cmap='gray')
            # Se pintan las detecciones del SAS:
            for r in regiones:
                if r.get_tag()=="{none}":
                    r.plot_region(ax, destacar=False)
            # Se pintan todas las regiones
            for r in regiones_filt:
                r.plot_region(ax, destacar=False)
            # Se destaca la región actual
            region.plot_region(ax, destacar=True)
            plt.show()
            
            while True:
                class_num = input("Class number: ")
                if class_num in ["","0","1","2","3","4"]:
                    break
                else:
                    print("Input inválido \n")
            if class_num=="":
                continue
            
            if region.get_forma() == "polygon":
                # Tupla con los parámetros de la región en formato YOLO:
                params = poly_2_bbox(region,int(class_num))
            elif region.get_forma() == "box":
                # Tupla con los parámetros de la región en formato YOLO:
                params = box_2_bbox(region,int(class_num),False)
            elif region.get_forma() == "circle":
                # Tupla con los parámetros de la región en formato YOLO:
                params = circle_2_bbox(region,int(class_num),False)
            elif region.get_forma() == "ellipse":
                # Tupla con los parámetros de la región en formato YOLO:
                params = ellipse_2_bbox(region,int(class_num),False)
                
            else:   
                params = None
            if params is not None:
                with open(anotaciones_path, "a", encoding='utf-8') as file:  
                    file.write(f"{params[0]} {params[1]} {params[2]} "\
                               f"{params[3]} {params[4]} {params[5]} "\
                               f"{params[6]} {params[7]} {params[8]} \n")
    else:
        for region in regiones_filt:   
            if region.get_forma() == "polygon":
                # Tupla con los parámetros de la región en formato YOLO:
                params = poly_2_bbox(region,0)
                            
            elif region.get_forma() == "box":
                # Tupla con los parámetros de la región en formato YOLO:
                params = box_2_bbox(region,0,True)
            
            elif region.get_forma() == "circle":
                # Tupla con los parámetros de la región en formato YOLO:
                params = circle_2_bbox(region,0,True)
            
            elif region.get_forma() == "ellipse":
                # Tupla con los parámetros de la región en formato YOLO:
                params = ellipse_2_bbox(region,0,True)
                
            else:   
                params = None
            
            if params is not None:
                with open(anotaciones_path, "a", encoding='utf-8') as file:  
                    file.write(f"{params[0]} {params[1]} {params[2]} "\
                               f"{params[3]} {params[4]} {params[5]} "\
                               f"{params[6]} {params[7]} {params[8]} \n")
                        
                        
def generar_YOLO(data_dir:str, class_dif:bool, draw_circles:bool=False, splits:int=3):
    """
    Función que aplica la función reg_2_YOLO() a todos los fits dentro del
    directorio que se pasa como parámetro. Los ficheros de regiones deben estar
    en el mismo directorio.

    Parameters
    ----------
    data_dir : str
        Directorio donde se encuentran las imágenes que se desean exportar en
        formato YOLO, junto con sus ficheros de regiones.
        
    class_diff : bool
        Indica si se quiere clasificar los objetos manualmente (True) o no (False).
        
    draw_circles : bool
        Indica si se quiere que las imágenes resultantes tengan superpuestos los
        círculos correspondientes a las regiones circulares sin tag.
        
    splits : int
        Número de divisiones para k-folding. Se crearán tantas carpetas alternando
        el set de validación como indique este argumento.

    Returns
    -------
    None.
    
    """
    # Creamos la carpeta YOLO:
    YOLO_dir = os.path.join(data_dir,"YOLO") # Directorio de destino
    try:
        if os.path.exists(YOLO_dir):
            # Si la carpeta ya existe, se borra junto con su contenido:
            shutil.rmtree(YOLO_dir)
        os.makedirs(YOLO_dir, exist_ok=True)
    except IOError:
        print("Imposible crear la carpeta "+YOLO_dir)
        sys.exit()
        
    # Obtenemos la lista de fits dentro de data_dir:
    nombres = os.listdir(data_dir)
    # Lista con los nombres de los archivos fits:
    lista_fits = [nombre for nombre in nombres if nombre.endswith('.fits')]
    array_fits = np.array(lista_fits)
    
    # Se aplica reg_2_YOLO a todas las imágenes de la lista train:
    for imagen in lista_fits:
        # Dirección del FIT en el disco, relativas al fichero .py:
        direccion_fit = os.path.join(data_dir, imagen)
        if draw_circles:
            reg_2_YOLO_circles(direccion_fit,class_dif)
        else:
            reg_2_YOLO(direccion_fit,class_dif)
            
    # k-fold para organizar los datos en splits carpetas diferentes.
    # Crear un objeto KFold con las divisiones que se hayan indicado
    kf = KFold(n_splits=splits, shuffle=True, random_state=42)
    
    # Mostrar las divisiones de k-folding
    for fold, (train_index, val_index) in enumerate(kf.split(array_fits)):
        # Creamos la carpeta para el fold:
        fold_dir=os.path.join(YOLO_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        # Creamos carpetas de images/labels, train/val:
        im_dir = os.path.join(fold_dir, "images")
        os.makedirs(im_dir, exist_ok=True)
        os.makedirs(os.path.join(im_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(im_dir, "validation"), exist_ok=True)
        
        lab_dir = os.path.join(fold_dir, "labels") 
        os.makedirs(lab_dir, exist_ok=True)
        os.makedirs(os.path.join(lab_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(lab_dir, "validation"), exist_ok=True)
        
        # Listas para las imágenes de validación y train:
        train_images = array_fits[train_index]
        val_images = array_fits[val_index]
        
        # Se copian las imágenes y anotaciones a las carpetas correspondienetes:
        for train_image in train_images:
            #imagen
            shutil.copy(os.path.join(YOLO_dir, os.path.splitext(train_image)[0]+".tif"), 
                        os.path.join(im_dir, "train", os.path.splitext(train_image)[0]+".tif"))
            #anotaciones:
            shutil.copy(os.path.join(YOLO_dir, os.path.splitext(train_image)[0]+".txt"), 
                        os.path.join(lab_dir, "train", os.path.splitext(train_image)[0]+".txt"))
            
        for val_image in val_images:
            #imagen
            shutil.copy(os.path.join(YOLO_dir, os.path.splitext(val_image)[0]+".tif"), 
                        os.path.join(im_dir, "validation", os.path.splitext(val_image)[0]+".tif"))
            #anotaciones:
            shutil.copy(os.path.join(YOLO_dir, os.path.splitext(val_image)[0]+".txt"), 
                        os.path.join(lab_dir, "validation", os.path.splitext(val_image)[0]+".txt"))