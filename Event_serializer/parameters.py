import datetime as dt
from numpy import array
#######################################################
####                 PATHS                         ####
#######################################################

raw_files_root_path = r'/mnt/compartido/d2025237/'              # Root path where raw files are located
processing_hub_path = r'/home/pc-igp-173/Documentos/EXP2025/processing_hub/'             # Path to where each raw file will be processed 
output_root_path = r'/home/pc-igp-173/Documentos/EXP2025/output/'                # Root path where output files will be saved

#######################################################
####                 PROCESSING PARAMETERS         ####
#######################################################

channels = [0, 1, 2]                       # Canales a analizar
cut = -20
decode= True                              # Activar decodificación. True: El archivo necesita ser decodificado.
code_vec= array([1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0])      # código. Expresar como una sola lista
nBaud=28                                        # Número de baudios del código

startDate=dt.date(2000,6,5)                # Fechas y horas limite para buscar eventos. Se sugiere dejar como está
endDate=dt.date(2050,6,5)
startTime=dt.time(0,0,0)
endTime=dt.time(23,59,59)

zoomed_time_size=90
zoomed_range_size=50

#######################################################
####                 FILTER PARAMETERS             ####
#######################################################

significance_filter_units='linear'  # 'linear' o 'dB', unidades del filtro de significancia
nSigma=5                            # Número de sigmas para el filtro de significancia
min_channels=3                      # Mínimo de canales requeridos en coincidencia
min_samples=3                       # Mínimo de muestras consecutivas requeridas en un mismo perfil

