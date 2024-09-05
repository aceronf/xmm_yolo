# -*- coding: utf-8 -*-
"""
Contiene varias funciones útiles para manejar y visualizar imágenes FITS y 
archivos de regiones, utilizando principalmente astropy y matplotlib.

@author: Alejandro Cerón Fernández
@date: 27/03/2024
"""
import os
from astropy.io import fits
from astropy.visualization import MinMaxInterval, AsymmetricPercentileInterval
from astropy.visualization import SqrtStretch
import matplotlib.pyplot as plt
import regiones as rg

def imreg_visor(reg_path:str, image_path:str, ax:plt.axes)->plt.axes:
    """
    Función que recibe como parámetros las rutas de una imagen FITS 
    y un archivo de regiones, y muestra ambos simultáneamente. Utiliza el 
    método plot_region de la clase Region y la función reg_lista. Devuelve la
    figura con la imagen y las regiones superpuestas.

    Parameters
    ----------
    reg_path : str
        Dirección al fichero de regiones en formato ds9.
    
    image_path : str
        Dirección a la imagen FITS.
        
    ax : plt.Axes
        Objeto Axes sobre el que se va a pintar la imagen con las regiones.

    Returns
    -------
    plt.axes
        Objeto axes con la imagen y las regiones superpuestas.

    """
    with fits.open(image_path) as lista_hdu:
        #Imagen:
        image = lista_hdu[0].data
        # Regiones:
        regiones = rg.reg_lista(reg_path)  
        
        # Normalización MinMax:
        interval_minmax = MinMaxInterval()
        minmax = interval_minmax(image)    
        # Combinación óptima de stretch y normalización por percentiles:
        comb = SqrtStretch() + AsymmetricPercentileInterval(0, 99.9)
        comb_im = comb(minmax)
        
        # Se representa la imagen:
        ax.imshow(comb_im, cmap='gray')
        ax.axis('off')
        # Las regiones se dibujan por encima
        for reg in regiones:
            try:
                reg.plot_region(ax, destacar=False)
            except ValueError:
                print(f"ValueError: Una región de tipo {reg.get_forma()} del archivo {reg_path} no pudo ser dibujada.")

    lista_hdu.close()
    return ax
    
def ver_todo(directorio:str, guardar=False):
    """
    Función que aplica imreg_visor() a todos los archivos fits de un directorio, 
    mostrando todos ellos con sus respectivas regiones (las regiones deben estar
    en el mismo directorio que las imágenes).

    Parameters
    ----------
    directorio : str
        Directorio donde se encuentran los FITS a visualizar y los correspondientes
        ficheros de regiones.
    
    guardar : bool
        Indica si se quiere guardar las imágenes en formato .png o no. Si sí, 
        se almacenan en una carpeta dentro de directorio llamada "visualizador".

    Returns
    -------
    None.

    """
    # Lista de nombres de archivos en la carpeta:
    nombres = os.listdir(directorio)
    # Lista con los nombres de los archivos fits:
    lista_fits = [nombre for nombre in nombres if 
                  (nombre.endswith('.fits') or nombre.endswith('.fit'))]
    if guardar:
        visualizador_dir = os.path.join(directorio, "visualizador")
        if not os.path.exists(visualizador_dir):
            os.makedirs(visualizador_dir)
    
    # Para todas las imágenes:
    for imagen in lista_fits:
        # Dirección de la imagen fits:
        direccion_fit = os.path.join(directorio, imagen)
        # Dirección del archivo de regiones correspondiente:
        reg_file = imagen[:11]+"_poly.reg"
        direccion_reg = os.path.join(directorio, reg_file)
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax=imreg_visor(direccion_reg,direccion_fit,ax)
        plt.show()
        if guardar:
            # Guardar la imagen en la carpeta "visualizador"
            nombre_imagen = os.path.splitext(imagen)[0] + '.png'
            direccion_imagen = os.path.join(visualizador_dir, nombre_imagen)
            fig.savefig(direccion_imagen, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            
        

