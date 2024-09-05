# -*- coding: utf-8 -*-
"""
Contiene el código para procesar ficheros de regiones en formato ds9
y generar una lista de todas las regiones contenidas en ellos, utilizando la
clase Region y la función reg_lista()
            
@author: Alejandro Cerón Fernández
@date: 18/03/2024
"""

import matplotlib.pyplot as plt
import matplotlib.patches as ptc

class Region:
    """
    Clase para crear regiones a partir de líneas en los ficheros de regiones.
    """
    def __init__(self, linea:str):
        """
        Crea una región a partir de una línea de texto.

        Parameters
        ----------
        line : str
            línea del fichero de regiones.

        Returns
        -------
        None.

        """
        #### forma:
        self.__forma:str = linea.split("(")[0]
        
        #### coordenadas y parámetros:
        parentesis = linea.split('(')[1].split(')')[0]
        self.__coordenadas:tuple = tuple(float(coord) for coord in parentesis.split(','))
        
        #### tags:
        if "tag=" in linea: 
            #man, sin o src
            self.__tag:str = linea.split('tag=')[1].split()[0]
        
        elif "color=white" in linea:
            # Las regiones blancas enmarcan detecciones que el sistema identifica
            # automáticamente como falsas. Guardamos esas regiones con la tag 
            # "{auto}"
            self.__tag:str = "{auto}"
            
        else:
            self.__tag:str = "{none}"
            
        #### color:
        if self.__tag == "{sin}":
            self.__color:str = "yellow"
            
        elif self.__tag == "{man}":
            self.__color:str = "red"
        
        elif self.__tag == "{src}":
            self.__color:str = "green"
            
        elif self.__tag == "{auto}":
            self.__color:str = "white"
        else:
            self.__color:str = "cyan"

    def get_forma(self)-> str:
        
        """
        Devuelve la forma de la región: box, circle, ellipse, polygon. 
        
        Returns
        -------
        str
            string con la forma.

        """
        return str(self.__forma)
    
    def get_tag(self)-> str:
        
        """
        Devuelve la tag de la región. 
        
        Returns
        -------
        str
            string con la tag, si la tiene. Si no la tiene devuelve "{none}".

        """
        return str(self.__tag)
    
    def get_color(self)-> str:
        
        """
        Devuelve el color de la región.
        
        Returns
        -------
        str
            string con el color.

        """
        return str(self.__color)
    
    def get_coordenadas(self)-> tuple:
        
        """
        Devuelve las coordenadas y parámetros que definen la región.
        
        Returns
        -------
        tuple
            tupla con las coordenadas.

        """
        return self.__coordenadas
    
    def plot_region(self, imagen, destacar:bool=False):
        """
        Pinta la región encima de una imagen (plot) que se pasa como parámetro.

        Parameters
        ----------
        imagen : matplotlib.axes.Axes
            Objeto Axes sobre el que se van a pintar las regiones.
            
        destacar : bool
            Indica si la región será destacada (linea más ancha) o no.
            
        Returns
        -------
        None.

        """
        if self.__forma == "circle":
            posx, posy, radio = self.__coordenadas
            circle = plt.Circle((posx, posy), radio, color=self.__color, fill=False, linewidth=(4 if destacar else 1))
            imagen.add_patch(circle)
            
        elif self.__forma == "box":
            posx, posy, ancho, alto, angle = self.__coordenadas         
            rect = plt.Rectangle((posx - ancho/2, posy - alto/2), ancho, alto, 
                                 angle=angle, edgecolor=self.__color, rotation_point="center",
                                 fill=False, linewidth=(4 if destacar else 1))
            imagen.add_patch(rect)
        
        elif self.__forma == "ellipse":
            posx, posy, radx, rady, angle = self.__coordenadas
            ellipse = ptc.Ellipse((posx, posy), 2*radx, 2*rady, angle=angle, edgecolor=self.__color, 
                                  fill=False, linewidth=(4 if destacar else 1))
            imagen.add_patch(ellipse)
        
        elif self.__forma == "polygon":
            vertices = [(float(self.__coordenadas[i]), float(self.__coordenadas[i+1])) 
                        for i in range(0, len(self.__coordenadas), 2)]
            polygon = ptc.Polygon(vertices, edgecolor=self.__color, fill=False, linewidth=(4 if destacar else 1))   
            imagen.add_patch(polygon)     
    
    
def reg_lista(reg_file_path:str)->list:
    """
    Función que recibe como parámetro la ruta a un archivo de regiones, lo
    lee y devuelve una lista con objetos de la clase Region.
    

    Parameters
    ----------
    reg_file_path : str
        dirección del archivo de regiones que será leído.

    Returns
    -------
    List
        lista con objetos de la clase Region.

    """
    regiones = [] # Lista donde almacenaremos regiones
    try:
        with open(reg_file_path, 'r') as reg_file:
            for line in reg_file:
                if "(" in line: 
                    regiones.append(Region(line))
                else:
                    continue
                    
    except IOError:
        print("No se encuentra el archivo de regiones "+reg_file_path)

    return regiones