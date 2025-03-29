import datetime as dt
###################################################################### GENERAL ########################################################################
path = r'/home/francisco/Documentos/JRO/data'

channels=[0,1,2]                           ## Canales a analizar
decode= True                               ## Activar decodificación. True: El archivo necesita ser decodificado.
code=[1,1,1,1,1,-1,-1,1,1,-1,1,-1,1]       ## código. Expresar como una sola lista
nBaud=13                                   ## Número de baudios del código

startDate=dt.date(2000,6,5)   ## Fechas y horas limite para buscar eventos. Se sugiere dejar como está
endDate=dt.date(2025,6,5)
startTime=dt.time(0,0,0)
endTime=dt.time(23,59,59)

profiles_lim= 20000    ## Número de perfiles máximo a leer a la vez para evitar llenar la memoria. Establecer en 1e20 para desactivar. Recomendado para archivos .r grandes
###################################################################### RTI_PLOT.py ########################################################################
num_ticks_y=25         ## Número de marcas en el eje Y del RTI
num_ticks_x=15         ## Número de marcas en el eje X del RTI

###################################################################### EVENT_IDENTIFIER.py ########################################################################
output_dir = r'/home/francisco/Documentos/JRO/trails'   ## Directorio de salida para los RTI amplificados de posibles eventos
exclude_range = [33.5,49]                               ## [min,max] Rango en km a ser excluido (Rango en el cual se observa electrochorro)


