import matplotlib 
matplotlib.use('Agg')
import numpy as np
from scipy.interpolate import interp1d
from schainpy.model import VoltageReader
import os
import shutil
import time as timelib
import pickle

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

def find_coincidences(significant_measurements_ls, min_channels=2):
    binary_matrices = [(mat != 0).astype(int) for mat in significant_measurements_ls]
    sum_matrix = sum(binary_matrices)
    coincidence_mat = (sum_matrix >= min_channels).astype(int)
    return coincidence_mat

def find_groups(arr, min_size=3):
    groups = []
    temp_group = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1] + 1:
            temp_group.append(arr[i])
        else:
            if len(temp_group) >= min_size:
                groups.append(temp_group)
            temp_group = [arr[i]]
    if len(temp_group) >= min_size:
        groups.append(temp_group)
    return groups

def find_trails(mat,min_samples,height_ls,exclude_range):
    up_lim=mat.shape[0]-1
    lo_lim=0
    trails=[]
    
    for j in range(1,mat.shape[1]-1):
        col_vec=mat[:,j]
        candidate_rows=np.where(col_vec > 0)[0]
        
        if len(candidate_rows)>=min_samples:
            possible_trails=find_groups(candidate_rows, min_size=min_samples)
    
            for possible_trail in possible_trails:
                includes_lo= possible_trail[0]==lo_lim
                includes_up= possible_trail[-1]==up_lim
                
                if includes_lo and includes_up:
                    side= possible_trail
                elif includes_lo:
                    side=possible_trail+[possible_trail[-1]+1]
                elif includes_up:
                    side=[possible_trail[0]-1]+possible_trail
                else:
                    side=[possible_trail[0]-1]+possible_trail+[possible_trail[-1]+1]
    
                left_clear= np.all(mat[side,j-1]==0)
                right_clear= np.all(mat[side,j+1]==0) 
                
                if left_clear and right_clear:
                    if not(height_ls[possible_trail[0]] > exclude_range[0] and height_ls[possible_trail[-1]] < exclude_range[1]):
                    #print('not in range')
                        trail= (possible_trail[0],possible_trail[-1],j)    ### (START,END,COLUMN)
                        trails.append(trail) 


    return trails

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

###################################################################### FILTER PARAMETERS ########################################################################
significance_filter_units='linear'
nSigma=5
min_channels=3
min_samples=3

###################################################################### FIGS INIT ########################################################################
figs_trails=[]
total_cr=0
trails_data=[]

###################################### MOD #################################33

with open("relation.pickle", "rb") as f:
    mod_relation = pickle.load(f)

mod_pickle=''
mod_dats=mod_relation[mod_pickle]

i=0
mod_crs=[]
with open(os.path.join('/home/pc-igp-173/Documentos/DATA/cosmic_rays/jun2023_oblique',mod_pickle), 'rb') as f:
    print(f'processing pickle')
    not_end=True
    while not_end:
        try:
            obj = pickle.load(f)
            mod_crs.append(obj)
            i+=1
        except EOFError as e:
            not_end=False
    print(f'total {i} crs')

mod_index=0
mod_index_max=len(mod_crs)

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

    #os.makedirs(output_dir, exist_ok=True) 
    ###################################################################### FILES ITERATION ########################################################################
    file_paths=[os.path.join(path, nombre) for nombre in mod_dats]
    file_paths=sorted(file_paths)#[7:]

    code_vec=np.asarray(code)

    
    for file in file_paths:
        tmp_file_path = shutil.move(file, tmp_dir_path)   ## Moving file to temporary location
        print(f'Moved {file} to {tmp_dir_path}')
        timelib.sleep(1)
        
        rawdataObj = VoltageReader()
        rawdataObj.name='VoltageReader'
        
        power=[[] for _ in channels]           ## Data structure initialization
        voltage=[[] for _ in channels]
        time=[]
        firsttime=True
        profile=0
        ###################################################################### SIGNAL CHAIN DATA COLLECTING ########################################################################    
        lastID=0
        trail_ID=0
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
                        complex_voltage=rawdataObj.dataOut.data[ch]
                        complex_voltage_decoded=np.correlate(complex_voltage, code_vec, mode='full')[:-nBaud+1] ## Decodification         #[:-len(complex_power)+1]#[:-nBaud+1]#[nBaud-1:]
                        power[ch].append((np.conj(complex_voltage_decoded)*complex_voltage_decoded).real.T)
                        voltage[ch].append(complex_voltage_decoded)

                        
                    if firsttime: ## Heights collection
                        height = rawdataObj.dataOut.heightList
                        firsttime=False
                else:
                    for ch in channels: ## Power collection
                        power[ch].append((np.conj(rawdataObj.dataOut.data[ch])*rawdataObj.dataOut.data[ch]).real.T)
                        voltage[ch].append(rawdataObj.dataOut.data[ch])
                        
                    if firsttime: ## Heights collection
                        height = rawdataObj.dataOut.heightList
                        firsttime=False

                if profile>=profiles_lim:
                    
                    print('Data reading completed')
                    
                    ## Data formating
                    for ch in channels:
                        power[ch] = np.column_stack(power[ch]).T
                        voltage[ch] = np.column_stack(voltage[ch])
                        

                    power_lin=np.array(power,dtype=np.float64)
                    power_db = 10*np.log10(power_lin+ 1)
                    
                    time_indices = np.linspace(0,len(time)-1,len(time),dtype=int)
                    change_ids = detect_timestamp_changes(time, time_indices)
                    change_ids =np.array(change_ids)
                    timestamps_change=np.array(time)[change_ids]
                    full_timestamps =  interpolate_timestamps(timestamps_change, change_ids, time_indices)

                    data =(full_timestamps,height,power_db,power_lin,voltage)

                    ###################################################################### NOISE VS HEIGHT ########################################################################    
                    for mod_cr in mod_crs:                        
                        if mod_cr['file'] == os.path.basename(file):
                            mod_time_ID=mod_cr['time_ID']
                            if data[0][mod_time_ID] == mod_cr['timestamp']:
                                mod_cr['mod_profile']=[volt_ch[:,mod_time_ID] for volt_ch in voltage]
                                mod_cr['mod_profile_prev']=[volt_ch[:,mod_time_ID-1] for volt_ch in voltage]
                                mod_cr['mod_profile_next']=[volt_ch[:,mod_time_ID+1] for volt_ch in voltage]

                                with open(os.path.join('/home/pc-igp-173/Documentos/DATA/cosmic_rays/jun2023_oblique','mod'+mod_pickle), 'ab') as f:
                                    pickle.dump(mod_cr, f)


                
                    profile=-1
                    power=[[] for _ in channels]           ## Data structure initialization
                    voltage=[[] for _ in channels]
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
            voltage[ch] = np.column_stack(voltage[ch])

        power_lin=np.array(power,dtype=np.float64)
        power_db = 10*np.log10(power_lin+ 1)
        
        time_indices = np.linspace(0,len(time)-1,len(time),dtype=int)
        change_ids = detect_timestamp_changes(time, time_indices)
        change_ids =np.array(change_ids)
        timestamps_change=np.array(time)[change_ids]
        full_timestamps =  interpolate_timestamps(timestamps_change, change_ids, time_indices)

        data =(full_timestamps,height,power_db,power_lin,voltage)

###################################################################### NOISE VS HEIGHT ########################################################################    
        for mod_cr in mod_crs:                        
            if mod_cr['file'] == os.path.basename(file):
                mod_time_ID=mod_cr['time_ID']
                if data[0][mod_time_ID] == mod_cr['timestamp']:
                    mod_cr['mod_profile']=[volt_ch[:,mod_time_ID] for volt_ch in voltage]
                    mod_cr['mod_profile_prev']=[volt_ch[:,mod_time_ID-1] for volt_ch in voltage]
                    mod_cr['mod_profile_next']=[volt_ch[:,mod_time_ID+1] for volt_ch in voltage]

                    with open(os.path.join('/home/pc-igp-173/Documentos/DATA/cosmic_rays/jun2023_oblique','mod'+mod_pickle), 'ab') as f:
                        pickle.dump(mod_cr, f)

        ###################################################################### DASHBOARD DISPLAYING ########################################################################    
        timelib.sleep(1)
        lastID=0
        shutil.move(tmp_file_path, path)
        timelib.sleep(1)  
        print('File processing completed')
    	
        del power, voltage, time, firsttime, profile, lastID, trail_ID
        del data
        del power_lin, power_db, time_indices, change_ids, timestamps_change, full_timestamps

        
    #print(f'{len(figs_trails)} possible events found')
    os.rmdir(os.path.join(path,'temp_reader'))

    print(f'TOTAL: {total_cr} events')
	
