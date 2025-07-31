import numpy as np
from scipy.interpolate import interp1d
from schainpy.model import VoltageReader
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import os
import shutil
import time as timelib
import matplotlib
matplotlib.use('TkAgg')  

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
        for name in os.listdir(path):
            if name.endswith(extension):
                full_path = os.path.join(path, name)
                filenames_list.append(name)
                filepaths_list.append(full_path)
    return filenames_list,filepaths_list

def detect_timestamp_changes(timestamps, ids):
    timestamps = np.array(timestamps)
    ids = np.array(ids)
    changes = np.where(timestamps[1:] != timestamps[:-1])[0] + 1  
    return (ids[0],*ids[changes])

def interpolate_timestamps(timestamps_change, ids_change, full_ids):
    timestamps_change = np.array(timestamps_change)
    ids_change = np.array(ids_change)
    
    interpolator = interp1d(ids_change, timestamps_change, kind='linear', fill_value="extrapolate")

    full_ids=np.asarray(full_ids)
    
    full_timestamps = interpolator(full_ids)
    
    # Extrapolate using the last interval's slope
    if len(ids_change) > 1:
        last_slope = (timestamps_change[-1] - timestamps_change[-2]) / (ids_change[-1] - ids_change[-2])
        last_idx = ids_change[-1]
        mask = full_ids >= last_idx
        full_timestamps[mask] = timestamps_change[-1] + last_slope * (full_ids[mask] - last_idx)
    
    return full_timestamps

###################################################################### DATA READING PARAMETERS ########################################################################

from parameters import *

if __name__ == '__main__':
    ###################################################################### TEMP DIR PREPARATION ########################################################################
    tmp_dir_path=os.path.join(path,'temp_reader/')
    try:
        os.makedirs(tmp_dir_path)
        print(f"Temporary directory '{tmp_dir_path}' created successfully.")
    except FileExistsError:
        for file in list_files(path=tmp_dir_path,extension='.r')[1]:
            shutil.move(file, path)
        print(f"Temporary directory '{tmp_dir_path}' already exists.")


    ###################################################################### FILES ITERATION ########################################################################
    file_paths=list_files(path=path,extension='.r')[1]
    file_paths=sorted(file_paths)#[7:]
    code_vec=np.asarray(code)

    for file in file_paths:
        tmp_file_path = shutil.move(file, tmp_dir_path)   ## Moving file to temporary location
        print(f'Moved {file} to {tmp_dir_path}')
        timelib.sleep(1)
        
        rawdataObj = VoltageReader()
        rawdataObj.name='VoltageReader'
        
        power=[[] for _ in channels]           ## Data structure initialization
        time=[]
        firsttime=True
        profile=0
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
                
                if decode:
                    for ch in channels: ## Power collection
                        complex_power=rawdataObj.dataOut.data[ch]
                        complex_power_decoded=np.correlate(complex_power, code_vec, mode='full')[:-nBaud+1] ## Decodification         #[:-len(complex_power)+1]#[:-nBaud+1]#[nBaud-1:]
                        power[ch].append((np.conj(complex_power_decoded)*complex_power_decoded).real.T)
                        
                    if firsttime: ## Heights collection
                        height = rawdataObj.dataOut.heightList
                        firsttime=False
                else:
                    for ch in channels: ## Power collection
                        power[ch].append((np.conj(rawdataObj.dataOut.data[ch])*rawdataObj.dataOut.data[ch]).real.T)
                        
                    if firsttime: ## Heights collection
                        height = rawdataObj.dataOut.heightList
                        firsttime=False

                if profile>=profiles_lim:
                    
                            ## Data formating
                    for ch in channels:
                        power[ch] = np.column_stack(power[ch]).T
                    power_lin=np.array(power,dtype=np.float64)
                    power_db = 10*np.log10(power_lin+ 1)
                    
                    time_indices = np.linspace(0,len(time)-1,len(time),dtype=int)
                    change_ids = detect_timestamp_changes(time, time_indices)
                    change_ids =np.array(change_ids)
                    timestamps_change=np.array(time)[change_ids]
                    full_timestamps =  interpolate_timestamps(timestamps_change, change_ids, time_indices)

                    data =(full_timestamps,height,power_db,power_lin)
                    height_ls=data[1]
                    time_indices=np.linspace(0,len(time)-1,len(time),dtype=int)
                    time_labels = [datetime.fromtimestamp(t).strftime("%H:%M") for t in data[0]]
                    time_dt_full = np.array([f'{datetime.fromtimestamp(t).strftime("%H:%M")}' for t,x_i in zip(data[0],time_indices)])
                    vmin, vmax = np.asarray(power_db).min(), np.asarray(power_db).max()
                    fig, axs = plt.subplots( 1,len(channels), figsize=(15, 15))
                    axs=axs.flatten()
                    x_ticks=time_dt_full         ## Ticks
                    y_ticks=height_ls
                    tick_idx_x = np.linspace(0, len(x_ticks) - 1, num_ticks_x, dtype=int)
                    tick_idx_y = np.linspace(0, len(y_ticks) - 1, num_ticks_y, dtype=int)
                    for ch in channels:
                        axs[ch].imshow(power_db[ch].T, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
                        axs[ch].set_title(f'Channel: {ch}')
                        axs[ch].set_xticks(tick_idx_x)
                        axs[ch].set_xticklabels([x_ticks[i] for i in tick_idx_x], rotation=90)
                        axs[ch].set_yticks(tick_idx_y)
                        axs[ch].set_yticklabels([f'{y_ticks[i]:.2f}' for i in tick_idx_y]) 
                        axs[ch].set_xlabel('Time (HH:MM)')
                        axs[ch].set_ylabel('Range (km)')

                    
                    plt.tight_layout()
                    plt.show() 

                    profile=-1
                    power=[[] for _ in channels]           ## Data structure initialization
                    time=[]
            
                profile+=1

            except Exception as e: 
                print(e)
                
                break
        
        del rawdataObj

        print('Data reading completed')
        
        ## Data formating
        for ch in channels:
            power[ch] = np.column_stack(power[ch]).T
        power_lin=np.array(power,dtype=np.float64)
        power_db = 10*np.log10(power_lin+ 1)
        
        time_indices = np.linspace(0,len(time)-1,len(time),dtype=int)
        change_ids = detect_timestamp_changes(time, time_indices)
        change_ids =np.array(change_ids)
        timestamps_change=np.array(time)[change_ids]
        full_timestamps =  interpolate_timestamps(timestamps_change, change_ids, time_indices)

        data =(full_timestamps,height,power_db,power_lin)
        height_ls=data[1]
        time_indices=np.linspace(0,len(time)-1,len(time),dtype=int)
        time_labels = [datetime.fromtimestamp(t).strftime("%H:%M") for t in data[0]]
        time_dt_full = np.array([f'{datetime.fromtimestamp(t).strftime("%H:%M")}' for t,x_i in zip(data[0],time_indices)])
        vmin, vmax = np.asarray(power_db).min(), np.asarray(power_db).max()
        fig, axs = plt.subplots( 1,len(channels), figsize=(15, 15))
        axs=axs.flatten()
        x_ticks=time_dt_full         ## Ticks
        y_ticks=height_ls
        tick_idx_x = np.linspace(0, len(x_ticks) - 1, num_ticks_x, dtype=int)
        tick_idx_y = np.linspace(0, len(y_ticks) - 1, num_ticks_y, dtype=int)
        for ch in channels:
            axs[ch].imshow(power_db[ch].T, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
            axs[ch].set_title(f'Channel: {ch}')
            axs[ch].set_xticks(tick_idx_x)
            axs[ch].set_xticklabels([x_ticks[i] for i in tick_idx_x], rotation=90)
            axs[ch].set_yticks(tick_idx_y)
            axs[ch].set_yticklabels([f'{y_ticks[i]:.2f}' for i in tick_idx_y]) 
            axs[ch].set_xlabel('Time (HH:MM)')
            axs[ch].set_ylabel('Range (km)')
        plt.tight_layout()
        plt.show() 

        ###################################################################### DASHBOARD DISPLAYING ########################################################################    
        timelib.sleep(1)
        shutil.move(tmp_file_path, path)
        timelib.sleep(1)  
        print('File processing completed')
    
    os.rmdir(os.path.join(path,'temp_reader'))
