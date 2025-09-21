from schainpy.model import VoltageReader
import datetime as dt
import os
import shutil
import time as timelib
import io
import sys
from datetime import datetime

###################################################################### FUNCTIONS ########################################################################
def list_files(path=None,extension=None):
    if path is None:
        path=os.getcwd()
    filenames_list = []
    filepaths_list = []
    if extension is None:
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            filenames_list.append(name)
            filepaths_list.append(full_path)
    else:
        n=len(extension)
        for name in os.listdir(path):
            if name.endswith(extension):
                full_path = os.path.join(path, name)
                filenames_list.append(name)
                filepaths_list.append(full_path)
    return filenames_list,filepaths_list

###################################################################### DATA READING PARAMETERS ########################################################################


path = "/home/pc-igp-173/Documentos/DATA/DATA_JUN_2024_Day2_coded/"
path = "/home/pc-igp-173/Documentos/DATA/temporal/"
startDate=dt.date(2000,6,5)
endDate=dt.date(2025,6,5)
startTime=dt.time(0,0,0)
endTime=dt.time(23,59,59)

###################################################################### TEMP DIR PREPARATION ########################################################################
tmp_dir_path=os.path.join(path,'temp_reader/')
try:
    os.mkdir(tmp_dir_path)
    print(f"Temporary directory '{tmp_dir_path}' created successfully.")
except FileExistsError:
    for file in list_files(path=tmp_dir_path,extension='.r')[1]:
        shutil.move(file, path)
    print(f"Temporary directory '{tmp_dir_path}' already exists.")

###################################################################### FILES ITERATION ########################################################################
file_paths=list_files(path=path,extension='.r')[1]
file_paths=sorted(file_paths)
corrupted_files=[]
time=[]
first_firsttime=True

for file in file_paths:
    tmp_file_path = shutil.move(file, tmp_dir_path)   ## Moving file to temporary location
    print(f'Moved {file} to {tmp_dir_path}')
    #timelib.sleep(0.01)
    
    rawdataObj = VoltageReader()
    rawdataObj.name='VoltageReader'
    

    firsttime=True

    ###################################################################### SIGNAL CHAIN DATA COLLECTING ########################################################################    
    while not(rawdataObj.flagNoMoreFiles):    
        try:
            rawdataObj.run(path = tmp_dir_path,
                        startDate=startDate,
                        endDate=endDate,
                        startTime=startTime,
                        endTime=endTime,
                        online=0,
                        walk=0,
                        expLabel='',
                        delay=5)
            
            time.append(rawdataObj.dataOut.utctime)   ## Timestamp collection
            
            if firsttime: ## Heights collection
                if first_firsttime:
                    buffer = io.StringIO()
                    sys.stdout = buffer 
                    rawdataObj.dataOut.systemHeaderObj.printInfo()
                    rawdataObj.dataOut.radarControllerHeaderObj.printInfo()
                    rawdataObj.dataOut.processingHeaderObj.printInfo()
                    sys.stdout = sys.__stdout__

                    first_firsttime=False
                height = rawdataObj.dataOut.heightList
                firsttime=False
        
        except Exception as e: 
            print(f'exception:{e}:')
            if f'{e}' != 'No more files to read':
                print(f'Error reading {file}')
                corrupted_files.append(os.path.basename(file))
            break
    else:
        del rawdataObj
        shutil.move(tmp_file_path, path)
        continue
    
    del rawdataObj
    print('Data reading completed')
    shutil.move(tmp_file_path, path)

os.rmdir(os.path.join(path,'temp_reader'))


print('\n'*4)
print('*/'*30)
print(f'Directory: {path}\n')
output = buffer.getvalue()
print('*/'*30)
print("HEADERS \n", output)
print('*/'*30)
time_0=datetime.fromtimestamp(time[0]).strftime("%H:%M")
time_0_mil=datetime.fromtimestamp(time[0]).strftime("%H%M")
time_last=datetime.fromtimestamp(time[-1]).strftime("%H:%M")
time_last_mil=datetime.fromtimestamp(time[-1]).strftime("%H%M")
print(f"TIMESTAMPS FROM {time_0} TO {time_last}\n")
print(f'Military time: {time_0_mil}-{time_last_mil}')
print('*/'*30)
print(f"RANGE FROM {height[0]} km TO {height[-1]} km\n")
print('*/'*30)
print(f"Corrupted files: {corrupted_files}\n")
print('*/'*30)
