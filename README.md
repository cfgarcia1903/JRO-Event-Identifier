# JRO-Event-Identifier
Scripts for detecting anomalous events in data from the main antenna of the Jicamarca Radio Observatory.

# Instrucciones de uso

## Descarga y dependencias
1)Descarga/clona el repositorio en un directorio de trabajo 
2)Utiliza los siguientes comandos en la terminal para crear un entorno de anaconda e instalar las dependencias necesarias para utilizar el programa
    ```
    conda create -n JRO_CR python=3.9.21
    conda activate JRO_CR
    pip install numpy==1.23.0
    pip install matplotlib==3.5.1
    pip install scipy==1.11.0
    pip install pandas==2.2.3
    pip install schainpy==3.0.1rc1
    
    ```
## rti_plot.py
### El script rti_plot.py permite graficar un RTI exploratorio de los datos, con el objetivo de identificar a qué rangos se observa el electrochorro, para poder excluir esos rangos de la búsqueda de eventos.

3)Abre el archivo parameters.py con cualquier editor de codigo. En la variable `path` Establece el directorio en el cual se encuentran los archivos raw del experimento a procesar. Los demás parámetros también pueden ser configurados. La sección `rti_plot.py` del archivo contiene parámetros específicos para ese script. Tras editar los parámetros, guarda los cambios. 
4)Abre una terminal en el directorio de trabajo y ejecuta:
```
    conda activate JRO_CR
    python rti_plot.py

```
5)Esto iniciará la ejecución del script. Tras unos segundos, se abrirá una ventana interactiva de matplotlib con los RTI de cada canal para unos cuantos perfiles analizados (Parámetro: `profiles_lim`). 
6)Anota el rango mínimo y máximo en el cual se observa electrochorro.
7)Si cierras la ventana, el script continuará analizando más perfiles y abrirá otra ventana con el RTI de los perfiles siguientes.
8)Puedes repetir el proceso hasta analizar toda la data. Para detenerlo, basta con usar `CTR + c` en la terminal.


## event_identifier.py
### Este script analiza la data perfil a perfil, encuentra posibles eventos anómalos, grafica RTI amplificados al rededor de estos eventos y los guarda en un directorio.

9) Abre el archivo parameters.py, en el parámetro `exclude_range` establece el rango mínimo y máximo en el cual hay electrochorro. Si no deseas usar esta opción, establece un valor alto como [1e20,2e20]
10) Configura el parámetro `output_dir` con la ruta al directorio donde deseas guardar los RTI amplificados de posibles eventos. 
11) Abre una terminal en el directorio de trabajo y ejecuta:
```
    conda activate JRO_CR
    python event_identifier.py

```
12) El programa tardará un tiempo en procesar los datos y a medida que encuentre eventos, irá almacenando los RTIs amplificados en el directorio establecido en el paso 10
