# -*- coding: utf-8 -*-
"""
Código para predecir sobre imágenes usando un modelo de detección de objetos de
YOLO entrenado previamente.

@author: Alejandro Cerón Fernández
"""
import sys
from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

def create_mask(boxes:list)->np.array:
    """
    Función que crea una máscara de la imagen a partir de una lista de bounding
    boxes que se pasa como parámetro.

    Parameters
    ----------
    boxes : list
        Lista de bounding boxes. Tuplas con formato (posx posy ancho alto)

    Returns
    -------
    mask : np.array
        Máscara aplicada a la imagen, correspondiente a los píxeles englobados
        por las bounding boxes.

    """
    # Create an empty mask
    mask = np.zeros((648,648), dtype=bool)
    
    # Draw filled rectangles on the mask for each bounding box
    for box in boxes:
        posx, posy, ancho, alto = box
        x1 = round(posx-ancho/2)
        x2 = round(posx+ancho/2)      
        y1 = round(posy-alto/2)
        y2 = round(posy+alto/2)
        mask[y1:y2, x1:x2] = True
    
    return mask

def calculate_iou(mask_orig, mask_pred)->tuple:
    """
    Función que calcula tres métricas para evaluar el funcionamiento de los
    modelos en cada imagen. Uno de ellos es la IoU (Intersection over union) 
    calculada con toda la superficie predicha por el modelo y toda la superficie
    marcada originalmente.

    Parameters
    ----------
    mask_orig : np.array
        máscara correspondiente a toda el área marcada originalmente.
    mask_pred : np.array
        máscara correspondiente a toda el área marcada por el modelo.

    Returns
    -------
    tuple
        IoU (intersection over union), IoO (intersection over original),
        IoP (intersection over predicted)

    """
    # Cálculo de la intersección y la unión
    intersection = np.logical_and(mask_orig, mask_pred)
    union = np.logical_or(mask_orig, mask_pred)
    
    # IoU:
    iou = np.sum(intersection) / np.sum(union)
    # IoO:
    ioo = np.sum(intersection) / np.sum(mask_orig)
    # IoP:
    if np.sum(mask_pred)==0:
        iop=0
    else:
        iop = np.sum(intersection) / np.sum(mask_pred)
        
    return iou, ioo, iop

def predictor(model_path:str, image:str, anns:str, save_images)->float:
    """
    Función que recibe un modelo y lo utiliza para predecir sobre una imagen
    que también se pasa como parámetro.
    

    Parameters
    ----------
    model_path : str
        dirección al modelo que se va a usar (.pt).
    image : str
        dirección a la imagen sobre la que se quiere predecir.
    anns : str
        dirección al fichero de texto con las anotaciones originales.
    save_images : bool
        True si se quieren guardar las imágenes en una carpeta dentro de runs/detect/train

    Returns
    -------
    tuple
        IoU, IoO, IoP

    """
    model = YOLO(model_path)
    result = model(image)[0] 
    # Lista de tuplas, cada una de ellas una predicción:
    lista_bbox_pred = [tuple(bbox) for bbox in np.array(result.boxes.xywh)]
    
    # Creamos un plot con 2 imágenes: la imagen con las bbox originales y la
    # imagen con las bbox predichas por YOLO
    fig_conjunta, ax_conjunta = plt.subplots(figsize=(16, 8), ncols=2, dpi=300)
    ax_conjunta[0].imshow(result.orig_img)
    ax_conjunta[1].imshow(result.orig_img)
    
    # Bounding boxes predichas por el modelo:
    for bbox in lista_bbox_pred:
        posx, posy, ancho, alto = bbox
        # Creamos el rectángulo matplotlib:
        rect = plt.Rectangle((posx - ancho/2, posy - alto/2), ancho, alto, 
                              edgecolor="green", fill=False, linewidth=2)
        ax_conjunta[0].add_patch(rect)
        
    # Bounding boxes originales:
    lista_bbox_orig=[]
    with open(anns, "r") as file:
        # Se lee cada línea del archivo
        for line in file:
            coords = [float(num)*648 for num in line.split()[1:]]
            original_bbox = tuple(coords)
            lista_bbox_orig.append(original_bbox)
            
    for bbox in lista_bbox_orig:
        posx, posy, ancho, alto = bbox
        # Creamos el rectángulo matplotlib:
        rect = plt.Rectangle((posx - ancho/2, posy - alto/2), ancho, alto, 
                              edgecolor="red", fill=False, linewidth=2)
        ax_conjunta[1].add_patch(rect)
    
    # Creamos máscaras:
    originales_mask = create_mask(lista_bbox_orig)
    predicciones_mask = create_mask(lista_bbox_pred)
    
    IoU, IoO, IoP = calculate_iou(originales_mask, predicciones_mask)
    # Incluimos estas métricas en las imágenes:
    # Utilizamos LaTeX:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')  
    plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{' \
                                           r'detect-family, ' \
                                           r'separate-uncertainty=true, ' \
                                           r'output-decimal-marker={.}, ' \
                                           r'exponent-product=\cdot, ' \
                                           r'inter-unit-product=\cdot, ' \
                                           r'}'
    text_box = rf'''$IoU={IoU:.2f}$
    $IoO={IoO:.2f}$
    $IoP={IoP:.2f}$'''
    ax_conjunta[0].text(0.05, 0.95, text_box, transform=ax_conjunta[0].transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=5), usetex=True)
    
    plt.show()
    
    ################# Se guardan las imágenes ########################
    if save_images:
        # Guardar imagen con predicciones del modelo (ax_conjunta[0])
        fig0, ax0 = plt.subplots()
        ax0.imshow(result.orig_img)
        for bbox in lista_bbox_pred:
            posx, posy, ancho, alto = bbox
            rect = plt.Rectangle((posx - ancho/2, posy - alto/2), ancho, alto, 
                                  edgecolor="green", fill=False, linewidth=2)
            ax0.add_patch(rect)
        ax0.text(0.05, 0.95, text_box, transform=ax0.transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=5), usetex=True)
        destino_pred = os.path.join(os.path.dirname(os.path.dirname(model_path)), "predicciones_val", 
                               os.path.basename(image)+"_pred.png")
        fig0.savefig(destino_pred, bbox_inches='tight', pad_inches=0, dpi=300)

        # Guardar imagen con bounding boxes originales (ax_conjunta[1])
        fig1, ax1 = plt.subplots()
        ax1.imshow(result.orig_img)
        for bbox in lista_bbox_orig:
            posx, posy, ancho, alto = bbox
            rect = plt.Rectangle((posx - ancho/2, posy - alto/2), ancho, alto, 
                                  edgecolor="red", fill=False, linewidth=2)
            ax1.add_patch(rect)
        destino_orig = os.path.join(os.path.dirname(os.path.dirname(model_path)), "predicciones_val", 
                               os.path.basename(image)+"_orig.png")
        fig1.savefig(destino_orig, bbox_inches='tight', pad_inches=0, dpi=300)

    return IoU, IoO, IoP


def predictor_classdif(model_path:str, image:str, anns:str, save_images:bool)->float:
    """
    Función que recibe un modelo (capaz de distinguir distintas clases de defectos) y lo 
    utiliza para predecir sobre una imagen que también se pasa como parámetro.
    

    Parameters
    ----------
    model_path : str
        dirección al modelo que se va a usar.
    image : str
        dirección a la imagen sobre la que se quiere predecir.
    anns : str
        dirección al fichero de texto con las anotaciones originales.
    save_images : bool
        True si se quieren guardar las imágenes en una carpeta dentro de runs/detect/train

    Returns
    -------
    tuple
        IoU, IoO, IoP

    """
    model = YOLO(model_path)
    result = model(image)[0] 
    # Lista de tuplas, cada una de ellas una predicción:
    lista_bbox_pred = [tuple(bbox) for bbox in np.array(result.boxes.xywh)]
    lista_class_pred = [int(objeto) for objeto in np.array(result.boxes.cls)]
    #Colores y nombres de las clases:
    colors = ["yellow", "blue", "green", "red"]
    names = ["Straylight", "OoT", "Extended", "CCD_Bands"]
    
    # Creamos un plot con la imagen original
    fig_conjunta, ax_conjunta = plt.subplots(figsize=(16, 8), ncols=2, dpi=300)
    ax_conjunta[0].set_title(os.path.basename(image)+" (Original)")
    ax_conjunta[0].imshow(result.orig_img)
    ax_conjunta[1].set_title(os.path.basename(image)+" (Predicción)")
    ax_conjunta[1].imshow(result.orig_img)
    
    # Bounding boxes predichas por el modelo:
    for index, bbox in enumerate(lista_bbox_pred):
        posx, posy, ancho, alto = bbox
        #Color y nombre de la bbox predicha:
        color = colors[lista_class_pred[index]]
        nombre = names[lista_class_pred[index]]
        # Creamos el rectángulo matplotlib:
        rect = plt.Rectangle((posx - ancho/2, posy - alto/2), ancho, alto, 
                              edgecolor=color, fill=False, linewidth=2)
        ax_conjunta[1].add_patch(rect)
        # Añadir texto encima del rectángulo
        text_x = posx - ancho/2 
        text_y = posy - alto/2  
        ax_conjunta[1].text(text_x, text_y, nombre, fontsize=10, color=color, ha='left', va='bottom')
        
    # Bounding boxes originales:
    lista_bbox_orig=[]
    lista_class_orig = []
    with open(anns, "r") as file:
        # Se lee cada línea del archivo
        for line in file:
            coords = [float(num)*648 for num in line.split()[1:]]
            original_bbox = tuple(coords)
            lista_bbox_orig.append(original_bbox)
            lista_class_orig.append(int(line.split()[0]))
            
    for index, bbox in enumerate(lista_bbox_orig):
        posx, posy, ancho, alto = bbox
        #Color y nombre de la bbox original:
        color = colors[lista_class_orig[index]]
        nombre = names[lista_class_orig[index]]
        # Creamos el rectángulo matplotlib:
        rect = plt.Rectangle((posx - ancho/2, posy - alto/2), ancho, alto, 
                              edgecolor=color, fill=False, linewidth=2)
        ax_conjunta[0].add_patch(rect)
        # Añadir texto encima del rectángulo
        text_x = posx - ancho/2  # x-coordinate of the text
        text_y = posy - alto/2  # y-coordinate of the text, adjust as needed
        ax_conjunta[0].text(text_x, text_y, nombre, fontsize=10, color=color, ha='left', va='bottom')
    
    # Creamos máscaras:
    originales_mask = create_mask(lista_bbox_orig)
    predicciones_mask = create_mask(lista_bbox_pred)
    
    IoU, IoO, IoP = calculate_iou(originales_mask, predicciones_mask)
    # Incluimos estas métricas en las imágenes, utilizando LaTeX:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')  
    plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx} \sisetup{' \
                                           r'detect-family, ' \
                                           r'separate-uncertainty=true, ' \
                                           r'output-decimal-marker={.}, ' \
                                           r'exponent-product=\cdot, ' \
                                           r'inter-unit-product=\cdot, ' \
                                           r'}'
    
    text_box = rf'''$IoU={IoU:.2f}$
    $IoO={IoO:.2f}$
    $IoP={IoP:.2f}$'''
    ax_conjunta[0].text(0.05, 0.95, text_box, transform=ax_conjunta[0].transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=5), usetex=True)
    
    plt.show()
    
    ################# Se guardan las imágenes ########################
    if save_images:
        # Guardar imagen con las predicciones del modelo (ax_conjunta[0])
        fig0, ax0 = plt.subplots()
        ax0.imshow(result.orig_img)
        for index, bbox in enumerate(lista_bbox_pred):
            posx, posy, ancho, alto = bbox
            #Color y nombre de la bbox predicha:
            color = colors[lista_class_pred[index]]
            nombre = names[lista_class_pred[index]]
            # Creamos el rectángulo matplotlib:
            rect = plt.Rectangle((posx - ancho/2, posy - alto/2), ancho, alto, 
                                  edgecolor=color, fill=False, linewidth=2)
            ax0.add_patch(rect)
            # Añadir texto encima del rectángulo
            text_x = posx - ancho/2
            text_y = posy - alto/2  
            ax0.text(text_x, text_y, nombre, fontsize=10, color=color, ha='left', va='bottom', usetex=True)
        ax0.text(0.05, 0.95, text_box, transform=ax0.transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=5), usetex=True)

        destino_pred = os.path.join(os.path.dirname(os.path.dirname(model_path)), "predicciones_val", 
                               os.path.basename(image)+"_pred.png")
        fig0.savefig(destino_pred, bbox_inches='tight', pad_inches=0, dpi=300)

        # Guardar imagen con las regiones originales (ax_conjunta[1])
        fig1, ax1 = plt.subplots()
        ax1.imshow(result.orig_img) #image
        for index, bbox in enumerate(lista_bbox_orig):
            posx, posy, ancho, alto = bbox
            #Color y nombre de la bbox original:
            color = colors[lista_class_orig[index]]
            nombre = names[lista_class_orig[index]]
            # Creamos el rectángulo matplotlib:
            rect = plt.Rectangle((posx - ancho/2, posy - alto/2), ancho, alto, 
                                  edgecolor=color, fill=False, linewidth=2)
            ax1.add_patch(rect)
            # Añadir texto encima del rectángulo
            text_x = posx - ancho/2  
            text_y = posy - alto/2  
            ax1.text(text_x, text_y, nombre, fontsize=10, color=color, ha='left', va='bottom', usetex=True)
            
        destino_orig = os.path.join(os.path.dirname(os.path.dirname(model_path)), "predicciones_val", 
                               os.path.basename(image)+"_orig.png")
        fig1.savefig(destino_orig, bbox_inches='tight', pad_inches=0, dpi=300)

    return IoU, IoO, IoP

def pred_todo(directorio:str, train:str, save_images:bool, classdif:bool):
    """
    Función que aplica predictor/predictor_classdif a todas las imágenes de validación del directorio
    que se pasa como parámetro. Al final, se muestran por pantalla los valores de IoU,
    IoO e IoP promediados con todas las imágenes.

    Parameters
    ----------
    directorio : str
        Carpeta YOLO donde están las fotos sobre las que se desea predecir.
        
    train : str
        Nombre de la carpeta dentro de "runs\detect" que tiene el modelo que
        se desea usar. Por ejemplo, "train", "train2"... Dentro, se buscará en la 
        carpeta weights y se escogerá el modelo best.pt. 
        
    save_images : bool
        True si se quieren guardar las imágenes en una carpeta dentro de runs/detect/train
    
    classdif : bool
        True si se quiere distinguir entre clases a la hora de representar las
        predicciones.

    Returns
    -------
    None.

    """
    # Lista de imágenes:
    images = os.listdir(os.path.join(directorio, "images", "validation"))
    
    # Dirección al modelo que se va a usar:
    model_path = os.path.join(directorio, "runs", "detect", train, "weights", "best.pt" )
    
    # Si se quieren guardar las imágenes como resultado, se crea una carpeta en runs/detect/train
    if save_images:
        dir_destino = os.path.join(directorio, "runs", "detect", train, "predicciones_val")
        if os.path.exists(dir_destino):
            shutil.rmtree(dir_destino)
        os.makedirs(dir_destino)
     
    # Iniciamos las métricas a 0:
    IoU_total=0
    funcionaIoU_total=0
    
    IoO_total=0
    funcionaIoO_total=0
    
    IoP_total=0
    funcionaIoP_total=0
    
    
    for tif in images:
        # Dirección a la imagen:
        image_path = os.path.join(directorio, "images", "validation", tif) 
        # Dirección a las anotaciones originales:
        ann_path = os.path.join(directorio, "labels", "validation", os.path.splitext(tif)[0]+".txt")
        
        if classdif:
            IoU, IoO, IoP = predictor_classdif(model_path, image_path, ann_path, save_images)
        else:
            IoU, IoO, IoP = predictor(model_path, image_path, ann_path, save_images)
            
        funcionaIoU = round(IoU)
        funcionaIoO = round(IoO)
        funcionaIoP = round(IoP)
        
        IoU_total += IoU
        funcionaIoU_total += funcionaIoU
        IoO_total += IoO
        funcionaIoO_total += funcionaIoO
        IoP_total += IoP
        funcionaIoP_total += funcionaIoP
    
    # Cálculo de las métricas promediadas:
    IoU_avg = IoU_total/len(images)
    IoO_avg = IoO_total/len(images)
    IoP_avg = IoP_total/len(images)
    funcionaIoU_avg = funcionaIoU_total/len(images)
    funcionaIoO_avg = funcionaIoO_total/len(images)
    funcionaIoP_avg = funcionaIoP_total/len(images)
    
    # Se muestran las métricas promediadas:
    print(f"IoU_avg = {IoU_avg}")   
    print(f"IoO_avg = {IoO_avg}")
    print(f"IoP_avg = {IoP_avg}")
    print(f"funcionaIoU_avg = {funcionaIoU_avg}")  
    print(f"funcionaIoO_avg = {funcionaIoO_avg}") 
    print(f"funcionaIoP_avg = {funcionaIoP_avg}")  
    



