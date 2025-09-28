# JRO-Event-Identifier 2.0.0
Sistema para la detección y procesamiento de eventos anómalos en datos de la antena principal del Observatorio de Radio Jicamarca.

# Instrucciones de uso - Event_serializer

## Descripción
Event_serializer es un software de procesamiento automatizado que analiza archivos de datos del radar de Jicamarca, separando las señales activas y pasivas, aplicando filtros de significancia y detectando eventos anómalos de manera eficiente.

## Instalación y dependencias

### 1. Descarga del repositorio
```bash
git clone https://github.com/cfgarcia1903/JRO-Event-Identifier.git
cd JRO-Event-Identifier
```

### 2. Configuración del entorno
```bash
conda create -n JRO_CR python=3.9.21
conda activate JRO_CR
pip install numpy==1.23.0
pip install matplotlib==3.5.1
pip install scipy==1.11.0
pip install pandas==2.2.3
pip install schainpy==3.0.1rc1
```

## Configuración

### 3. Configuración de parámetros
Edita el archivo `Event_serializer/parameters.py` con los siguientes parámetros:

#### Rutas principales:
- `raw_files_root_path`: Directorio donde se encuentran los archivos .r sin procesar
- `processing_hub_path`: Directorio temporal para procesamiento (se crea automáticamente en caso no exista)
- `output_root_path`: Directorio donde se guardarán los archivos procesados (.pickle)

#### Parámetros de procesamiento:
- `channels`: Lista de canales a analizar (ej: [0, 1, 2])
- `decode`: True si los datos necesitan decodificación
- `code_vec`: Vector de código para decodificación
- `nBaud`: Número de baudios del código
- `startDate` y `endDate`: Rango máximo de fechas para procesar
- `startTime` y `endTime`: Rango máximo de horas para procesar

#### Parámetros de filtrado:
- `nSigma`: Número de sigmas para filtro de significancia
- `significance_filter_units`: 'linear' o 'dB', escala sobre la que se aplica el filtro de significancia
- `min_channels`: Mínimo de canales requeridos en coincidencia
- `min_samples`: Mínimo de muestras consecutivas requeridas

## Ejecución

### 4. Ejecutar el procesador
```bash
conda activate JRO_CR
cd Event_serializer
python main.py
```
En caso se ejecute en un servidor, será necesario tener una interfaz gráfica para que se pueda cargar el módulo Signal Chain sin errores

## Funcionamiento del sistema

El software Event_serializer funciona de la siguiente manera:

1. **Preparación**: Crea un directorio temporal de procesamiento
2. **Listado**: Encuentra todos los archivos .r en el directorio de datos
3. **Procesamiento iterativo**: Para cada archivo:
   - Lo copia al directorio temporal
   - Separa los datos en señales activas y pasivas  
   - Aplica decodificación (si está habilitada)
   - Crea matrices RTI (Range-Time-Intensity) para cada tipo de señal
   - Aplica filtros de significancia, coincidencia y forma
   - Procesa y guarda los eventos detectados como archivos .pickle
   - Limpia los archivos temporales

### Archivos de salida
- Los eventos detectados se guardan como archivos .pickle en `output_root_path`
- Formato: `{nombre_archivo}_B{bloque}_active.pickle` y `{nombre_archivo}_B{bloque}_passive.pickle` dependiendo si los perfiles procesados cuentan con pulso de transmisión o si el radar se encuentra funcionando de forma pasiva
- Cada archivo contiene diccionarios con datos de los eventos anómalos identificados (un diccionario por evento)

### Monitoreo del progreso
El sistema proporciona información en tiempo real sobre:
- Archivos siendo procesados
- Bloques de datos analizados
- Eventos detectados y guardados


## Notas importantes

- El sistema procesa automáticamente todos los archivos .r encontrados en el directorio especificado
- Se recomienda tener suficiente espacio en disco para los archivos temporales y de salida
- Los archivos de salida son archivos binarios Python (.pickle) que pueden ser cargados posteriormente para análisis
- El procesamiento puede tomar tiempo considerable dependiendo del tamaño y cantidad de archivos 


