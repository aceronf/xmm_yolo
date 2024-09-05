# -*- coding: utf-8 -*-
"""
Contiene las funciones necesarias para filtrar las observaciones de acuerdo con
las regiones que contienen y sus tags.

@author: Alejandro Cerón Fernández
@date: 27/03/2024
"""
import os
import shutil
import sys
import matplotlib.pyplot as plt
import regiones as rg
from visualizador import imreg_visor


# Directorio con las imágenes fits:
directorio_im = "oimages_4xmm_image"
# Directorio con las regiones correspondientes:
directorio_reg = "polyreg_files_4xmm_image" 

# Lista de nombres de los archivos de regiones en la carpeta de regiones:
nombres_reg = os.listdir(directorio_reg)
# Lista de nombres de los archivos fit en la carpeta de imágenes:
nombres_im = os.listdir(directorio_im)

def save_polymanreg():
    """
    Guarda en una carpeta "muestra_1" las observaciones con alguna región de forma 
    poligonal y tag {man}.

    Returns
    -------
    None.

    """
    # Creamos la carpeta muestra_1:
    folder_name = "muestra_1"
    try:
        if os.path.exists(folder_name):
            # Si la carpeta ya existe, se borra junto con su contenido:
            shutil.rmtree(folder_name)
        os.makedirs(folder_name, exist_ok=True)
    except IOError:
        print(f"Imposible crear la carpeta {folder_name}")
        sys.exit()
    
    for nombre in nombres_reg:
        # Dirección del archivo de regiones relativa a este .py:
        direccion_reg = os.path.join(directorio_reg, nombre)
        # Se obtiene la lista de regiones en el archivo:
        try:
            lista = rg.reg_lista(direccion_reg)
        except IndexError:
            return
        # Booleano que es True si hay alguna región que cumpla la condición:
        hay_polyman = any((x.get_forma()=="polygon" and x.get_tag()=="{man}") for x in lista)
        
        # Si el booleano se cumple, guardamos en el directorio "muestra_1" el
        # fits con su archivo de regiones:
        if hay_polyman and (nombre[:11]+"EPX000OIMAGE8000.fits" in nombres_im):
            # Guardamos el archivo de regiones:
            try:
                shutil.copyfile(direccion_reg, os.path.join(folder_name, nombre))
            except IOError:
                print("No se ha podido copiar el archivo "+nombre)
                
            # Guardamos la imagen fits:
            nombre_fit = nombre[:11]+"EPX000OIMAGE8000.fits"
            origen_fit = os.path.join(directorio_im, nombre_fit)
            destino_fit = os.path.join(folder_name, nombre_fit)
            try:
                shutil.copyfile(origen_fit, destino_fit)
            except IOError:
                print("No se ha podido copiar el archivo "+nombre_fit)
                
def save_boxmanreg():
    """
    Guarda en una carpeta "muestra_2" las observaciones con alguna región de forma 
    box y tag {man}, con un tamaño mínimo de 150 píxeles en alguna de las dos 
    dimensiones.

    Returns
    -------
    None.
    
    """
    # Creamos la carpeta muestra_2:
    folder_name = "muestra_2"
    try:
        if os.path.exists(folder_name):
            # Si la carpeta ya existe, se borra junto con su contenido:
            shutil.rmtree(folder_name)
        os.makedirs(folder_name, exist_ok=True)
    except IOError:
        print(f"Imposible crear la carpeta {folder_name}")
        sys.exit()
    
    for nombre in nombres_reg:
        # Dirección del archivo de regiones relativa a este .py:
        direccion_reg = os.path.join(directorio_reg, nombre)
        # Se obtiene la lista de regiones en el archivo:
        try:
            lista = rg.reg_lista(direccion_reg)
        except IndexError:
            return
        # Booleano que es True si hay alguna región box con tag man y con
        # altura o anchura mayor de 150:
        hay_boxman = any((x.get_forma()=="box" and 
                          x.get_tag()=="{man}" and
                          (x.get_coordenadas()[2]>150 or x.get_coordenadas()[3]>150)
                          ) for x in lista)
        
        # Si el booleano se cumple, guardamos en el directorio "muestra_2" el
        # fits con su archivo de regiones:
        if hay_boxman and (nombre[:11]+"EPX000OIMAGE8000.fits" in nombres_im):
            # Guardamos el archivo de regiones:
            try:
                shutil.copyfile(direccion_reg, os.path.join(folder_name, nombre))
            except IOError:
                print("No se ha podido copiar el archivo "+nombre)
                
            # Guardamos el archivo fits:
            nombre_fit = nombre[:11]+"EPX000OIMAGE8000.fits"
            origen_fit = os.path.join(directorio_im, nombre_fit)
            destino_fit = os.path.join(folder_name, nombre_fit)
            try:
                shutil.copyfile(origen_fit, destino_fit)
            except IOError:
                print("No se ha podido copiar el archivo "+nombre_fit)
    
def save_sinreg():
    """
    Guarda en una carpeta "muestra_3" las observaciones con alguna región de 
    tag {sin} y ninguna de tag {man}.

    Returns
    -------
    None.
    
    """
    # Creamos la carpeta muestra_3:
    folder_name = "muestra_3"
    try:
        if os.path.exists(folder_name):
            # Si la carpeta ya existe, se borra junto con su contenido:
            shutil.rmtree(folder_name)
        os.makedirs(folder_name, exist_ok=True)
    except IOError:
        print(f"Imposible crear la carpeta {folder_name}")
        sys.exit()
    
    for nombre in nombres_reg:
        # Dirección del archivo de regiones relativa a este .py:
        direccion_reg = os.path.join(directorio_reg, nombre)
        # Se obtiene la lista de regiones en el archivo:
        try:
            lista = rg.reg_lista(direccion_reg)
        except IndexError:
            return
        # Booleano que es True si hay alguna región con tag sin y ninguna con man
        hay_sin = any(x.get_tag()=="{sin}" for x in lista) and not any(x.get_tag()=="{man}" for x in lista)
        
        # Si el booleano se cumple, guardamos en el directorio "muestra_3" el
        # fits con su archivo de regiones:
        if hay_sin and (nombre[:11]+"EPX000OIMAGE8000.fits" in nombres_im):
            # Guardamos el archivo de regiones:
            try:
                shutil.copyfile(direccion_reg, os.path.join(folder_name, nombre))
            except IOError:
                print("No se ha podido copiar el archivo "+nombre)
                
            # Guardamos el archivo fits:
            nombre_fit = nombre[:11]+"EPX000OIMAGE8000.fits"
            origen_fit = os.path.join(directorio_im, nombre_fit)
            destino_fit = os.path.join(folder_name, nombre_fit)
            try:
                shutil.copyfile(origen_fit, destino_fit)
            except IOError:
                print("No se ha podido copiar el archivo "+nombre_fit)
    
def save_okreg():
    """
    Guarda en una carpeta "muestra_0" las observaciones que no tienen regiones
    con tag {sin} ni {man}.

    Returns
    -------
    None.

    """
    # Creamos la carpeta muestra_0:
    folder_name = "muestra_0"
    try:
        if os.path.exists(folder_name):
            # Si la carpeta ya existe, se borra junto con su contenido:
            shutil.rmtree(folder_name)
        os.makedirs(folder_name, exist_ok=True)
    except IOError:
        print(f"Imposible crear la carpeta {folder_name}")
        sys.exit()
    
    for nombre in nombres_reg:
        # Dirección del archivo de regiones relativa a este .py:
        direccion_reg = os.path.join(directorio_reg, nombre)
        # Se obtiene la lista de regiones en el archivo:
        try:
            lista = rg.reg_lista(direccion_reg)
        except IndexError:
            return
        # Booleano que es True si no hay ninguna región con tag sin ni man:
        no_man_sin = not any((x.get_tag()=="{man}" or
                           x.get_tag()=="{sin}") for x in lista)
        
        # Si el booleano se cumple, guardamos en el directorio "muestra_0" el
        # fits con su archivo de regiones:
        if no_man_sin and (nombre[:11]+"EPX000OIMAGE8000.fits" in nombres_im):
            # Guardamos el archivo de regiones:
            try:
                shutil.copyfile(direccion_reg, os.path.join(folder_name, nombre))
            except IOError:
                print("No se ha podido copiar el archivo "+nombre)
                
            # Guardamos el archivo fits:
            nombre_fit = nombre[:11]+"EPX000OIMAGE8000.fits"
            origen_fit = os.path.join(directorio_im, nombre_fit)
            destino_fit = os.path.join(folder_name, nombre_fit)
            try:
                shutil.copyfile(origen_fit, destino_fit)
            except IOError:
                print("No se ha podido copiar el archivo "+nombre_fit)
                
def manual_filter(folder:str):
    """
    Función que muestra una por una las imágenes fits contenidas en "folder" con
    sus regiones, y pregunta por teclado si el usuario quiere conservarla o
    eliminarla (y/n).

    Parameters
    ----------
    folder : str
        Carpeta que se quiere filtrar manualmente.

    Returns
    -------
    None.

    """
    # Obtenemos la lista de archivos dentro de data_dir:
    nombres = os.listdir(folder)
    # Lista con los nombres de los archivos fits:
    lista_fits = [nombre for nombre in nombres if nombre.endswith('.fits')]
    # Recorremos la lista:
    for nombre_fit in lista_fits:
        nombre_reg = nombre_fit[:11]+"_poly.reg" # Nombre del archivo de regiones
        # Direcciones del fits y del archivo de regiones:
        fit_path = os.path.join(folder, nombre_fit)
        reg_path = os.path.join(folder, nombre_reg)
        # Se representa la imagen:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0, 0, 1, 1])
        ax=imreg_visor(reg_path, fit_path, ax)
        plt.show()
        # Se pregunta si se desea conservar la imagen o no:
        while True:
            decision = input("¿Conservar imagen? (Y/N): ").strip().lower()

            if decision == 'y' or decision == 'n':
                break
            else:
                print("Input inválido \n")

        if decision == 'y':
            # Conservar imagen
            continue
        elif decision == 'n':
            # Eliminar imagen
             os.remove(fit_path)
             os.remove(reg_path)