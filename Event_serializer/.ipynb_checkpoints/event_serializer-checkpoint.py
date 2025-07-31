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
                    
                    height_ls=data[1]
                    if significance_filter_units=='linear':
                        power_matrices=data[3]
                    elif significance_filter_units=='dB':
                        power_matrices=data[2]
                    
                    for i, channel in enumerate(channels):
                        mean_pow_ls=[]
                        sd_pow_ls=[]
                        max_ls=[]
                        for h_index in range(len(height_ls)):
                            pow_vec=power_matrices[channel].T[h_index]
                            mean_pow_ls.append(np.mean(pow_vec))
                            sd_pow_ls.append(np.std(pow_vec))
                            max_ls.append(np.max(pow_vec))
                        mean_pow_ls=np.asarray(mean_pow_ls)
                        sd_pow_ls=np.asarray(sd_pow_ls)
                        max_ls=np.asarray(max_ls)
                        significance_thr_ls= mean_pow_ls+nSigma*sd_pow_ls
                    
                        if significance_filter_units=='linear': ##transform everything into dB
                            mean_pow_ls=10*np.log10(mean_pow_ls)
                            max_ls=10*np.log10(max_ls)
                            significance_thr_ls= 10*np.log10(significance_thr_ls)
                    
                    print('Noise profile completed')
                    

                    ###################################################################### FIND SIGNIFICANT EVENTS PER CHANNEL ########################################################################    

                    significant_measurements_ls=[]

                    for ch in channels:
                        mean_pow_ls=[]
                        sd_pow_ls=[]
                        for h_index in range(len(height_ls)):
                            pow_vec=power_matrices[ch].T[h_index]
                            mean_pow_ls.append(np.mean(pow_vec))
                            sd_pow_ls.append(np.std(pow_vec))
                        mean_pow_ls=np.asarray(mean_pow_ls)
                        sd_pow_ls=np.asarray(sd_pow_ls)
                        significance_thr_ls= mean_pow_ls+nSigma*sd_pow_ls
                        
                        mat_significance_thr_ch= np.repeat(significance_thr_ls, power_matrices[ch].T.shape[1], axis=0).reshape(power_matrices[ch].T.shape)
                        significant_measurements_ch= np.where(power_matrices[ch].T> mat_significance_thr_ch, power_matrices[ch].T, 0)
                        significant_measurements_ls.append(significant_measurements_ch)

                    ###################################################################### FIND COINCIDENCES ########################################################################    
                    
                    coincidence_mat=find_coincidences(significant_measurements_ls, min_channels=min_channels)
                    
                    channels_data_lin=[]
                    for n in channels:    ## Combined linear power
                        linear=data[3][n].T
                        channels_data_lin.append(linear)
                    joint_data_linear=sum(channels_data_lin)
                    joint_data_dB=10*np.log10(joint_data_linear+1)
                    
                    joint_significant_coincident_dB= coincidence_mat * joint_data_dB
                    ###################################################################### FIND TRAILS ########################################################################    
                    
                    trails=find_trails(joint_significant_coincident_dB,min_samples,height_ls,exclude_range)
                    print(f'{len(trails)} possible events were found')
                    if len(trails):
                        mat= joint_data_dB

                        vmin, vmax = mat.min(), mat.max()  
                        big_mat_size=len(data[0]),len(height_ls)

                        
                        for trail_ID_mini,trail in enumerate(trails):
                            trail_ID = lastID + trail_ID_mini

                            i_size=zoomed_time_size
                            j_size=zoomed_range_size

                            i_mid,j_mid= trail[2],int((trail[0]+trail[1])/2)


                            if i_mid-i_size//2 < 0:
                                i_min=0
                                i_max=i_min+i_size
                            elif i_mid+i_size//2 > big_mat_size[0]-1:
                                i_max=big_mat_size[0]-1
                                i_min=i_max-i_size
                            else:
                                i_min=i_mid-i_size//2
                                i_max=i_min+i_size

                            if j_mid-j_size//2 < 0:
                                j_min=0
                                j_max=j_min+j_size
                            elif j_mid+j_size//2 > big_mat_size[1]-1:
                                j_max=big_mat_size[1]-1
                                j_min=j_max-j_size
                            else:
                                j_min=j_mid-j_size//2
                                j_max=j_min+j_size


                            zoomed_mat=mat[j_min:j_max,i_min:i_max]
                            #print(voltage[0].shape)
                            zoomed_voltages=[volt_ch[j_min:j_max,i_min:i_max] for volt_ch in voltage]
                            voltage_profile=[volt_ch[:,trail[2]] for volt_ch in voltage]

                            trail_power_db=joint_data_dB[trail[0]:trail[1]+1,trail[2]]
                            trail_power_lin=joint_data_linear[trail[0]:trail[1]+1,trail[2]]

                            trail_data={'time_ID':trail[2],'timestamp':data[0][trail[2]],
                                    'Range_start_ID':trail[0],'Range_end_ID':trail[1],
                                    'Range_start':height_ls[trail[0]],'Range_end':height_ls[trail[1]],
                                    'Power_dB':trail_power_db,'Power_lin':trail_power_lin,
                                    'Matrix': zoomed_mat,
                                    'Norm_Matrix': (zoomed_mat-vmin)/(vmax-vmin),
                                    'Volt_Matrices':zoomed_voltages,
                                    'Volt_profile':voltage_profile,
                                    'CosmicRay': None,
                                    'file':os.path.basename(file)
                                    }
                            
                            #trails_data.append(trail_data)

                            with open(output_path_pickle, 'ab') as f:
                                pickle.dump(trail_data, f)
                            total_cr+=1
                            #with open(output_path_pickle, 'wb') as f:
                            #    pickle.dump(trails_data, f)

                        lastID = trail_ID+1
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
        
        height_ls=data[1]
        if significance_filter_units=='linear':
            power_matrices=data[3]
        elif significance_filter_units=='dB':
            power_matrices=data[2]
        
        for i, channel in enumerate(channels):
            mean_pow_ls=[]
            sd_pow_ls=[]
            max_ls=[]
            for h_index in range(len(height_ls)):
                pow_vec=power_matrices[channel].T[h_index]
                mean_pow_ls.append(np.mean(pow_vec))
                sd_pow_ls.append(np.std(pow_vec))
                max_ls.append(np.max(pow_vec))
            mean_pow_ls=np.asarray(mean_pow_ls)
            sd_pow_ls=np.asarray(sd_pow_ls)
            max_ls=np.asarray(max_ls)
            significance_thr_ls= mean_pow_ls+nSigma*sd_pow_ls
        
            if significance_filter_units=='linear': ##transform everything into dB
                mean_pow_ls=10*np.log10(mean_pow_ls)
                max_ls=10*np.log10(max_ls)
                significance_thr_ls= 10*np.log10(significance_thr_ls)
        
        print('Noise profile completed')
        

        ###################################################################### FIND SIGNIFICANT EVENTS PER CHANNEL ########################################################################    

        significant_measurements_ls=[]

        for ch in channels:
            mean_pow_ls=[]
            sd_pow_ls=[]
            for h_index in range(len(height_ls)):
                pow_vec=power_matrices[ch].T[h_index]
                mean_pow_ls.append(np.mean(pow_vec))
                sd_pow_ls.append(np.std(pow_vec))
            mean_pow_ls=np.asarray(mean_pow_ls)
            sd_pow_ls=np.asarray(sd_pow_ls)
            significance_thr_ls= mean_pow_ls+nSigma*sd_pow_ls
            
            mat_significance_thr_ch= np.repeat(significance_thr_ls, power_matrices[ch].T.shape[1], axis=0).reshape(power_matrices[ch].T.shape)
            significant_measurements_ch= np.where(power_matrices[ch].T> mat_significance_thr_ch, power_matrices[ch].T, 0)
            significant_measurements_ls.append(significant_measurements_ch)

        ###################################################################### FIND COINCIDENCES ########################################################################    
        
        coincidence_mat=find_coincidences(significant_measurements_ls, min_channels=min_channels)
        
        channels_data_lin=[]
        for n in channels:    ## Combined linear power
            linear=data[3][n].T
            channels_data_lin.append(linear)
        joint_data_linear=sum(channels_data_lin)
        joint_data_dB=10*np.log10(joint_data_linear+1)
        
        joint_significant_coincident_dB= coincidence_mat * joint_data_dB
        ###################################################################### FIND TRAILS ########################################################################    

        zoom_time= 45   #number of pixels in each direction
        zoom_range= 25 

        trails=find_trails(joint_significant_coincident_dB,min_samples,height_ls,exclude_range)

        if len(trails):
            print(f'{len(trails)} possible events were found')
            mat= joint_data_dB

            vmin, vmax = mat.min(), mat.max()  

            big_mat_size=len(data[0]),len(height_ls)

            
            for trail_ID_mini,trail in enumerate(trails):
                trail_ID = lastID + trail_ID_mini
                i_size=zoomed_time_size
                j_size=zoomed_range_size

                i_mid,j_mid= trail[2],int((trail[0]+trail[1])/2)


                if i_mid-i_size//2 < 0:
                    i_min=0
                    i_max=i_min+i_size
                elif i_mid+i_size//2 > big_mat_size[0]-1:
                    i_max=big_mat_size[0]-1
                    i_min=i_max-i_size
                else:
                    i_min=i_mid-i_size//2
                    i_max=i_min+i_size

                if j_mid-j_size//2 < 0:
                    j_min=0
                    j_max=j_min+j_size
                elif j_mid+j_size//2 > big_mat_size[1]-1:
                    j_max=big_mat_size[1]-1
                    j_min=j_max-j_size
                else:
                    j_min=j_mid-j_size//2
                    j_max=j_min+j_size

                
                zoomed_mat=mat[j_min:j_max,i_min:i_max]
                zoomed_voltages=[volt_ch[j_min:j_max,i_min:i_max] for volt_ch in voltage]
                voltage_profile=[volt_ch[:,trail[2]] for volt_ch in voltage]

                trail_power_db=joint_data_dB[trail[0]:trail[1]+1,trail[2]]
                trail_power_lin=joint_data_linear[trail[0]:trail[1]+1,trail[2]]

                trail_data={'time_ID':trail[2],'timestamp':data[0][trail[2]],
                        'Range_start_ID':trail[0],'Range_end_ID':trail[1],
                        'Range_start':height_ls[trail[0]],'Range_end':height_ls[trail[1]],
                        'Power_dB':trail_power_db,'Power_lin':trail_power_lin,
                        'Matrix': zoomed_mat,
                        'Norm_Matrix': (zoomed_mat-vmin)/(vmax-vmin),
                        'Volt_Matrices':zoomed_voltages,
                        'Volt_profile':voltage_profile,
                        'CosmicRay': None,
                        'file':os.path.basename(file)
                        }
                
                #trails_data.append(trail_data)
                
                with open(output_path_pickle, 'ab') as f:
                    pickle.dump(trail_data, f)
                total_cr+=1
                #with open(output_path_pickle, 'wb') as f:
                #    pickle.dump(trails_data, f)


        ###################################################################### DASHBOARD DISPLAYING ########################################################################    
        timelib.sleep(1)
        lastID=0
        shutil.move(tmp_file_path, path)
        timelib.sleep(1)  
        print('File processing completed')
    	
        del power, voltage, time, firsttime, profile, lastID, trail_ID
        del data, power_matrices, pow_vec,mat_significance_thr_ch,significant_measurements_ch
        del channels_data_lin,linear,joint_data_linear,joint_data_dB,joint_significant_coincident_dB
        del power_lin, power_db, time_indices, change_ids, timestamps_change, full_timestamps
        del height_ls, mean_pow_ls, sd_pow_ls, max_ls, significance_thr_ls
        del significant_measurements_ls, coincidence_mat
        try:
            del zoomed_mat, zoomed_voltages, voltage_profile
            del trail_power_db, trail_power_lin
            del trail_data
        except:
            pass
        
    #print(f'{len(figs_trails)} possible events found')
    os.rmdir(os.path.join(path,'temp_reader'))

    print(f'TOTAL: {total_cr} events')
	
