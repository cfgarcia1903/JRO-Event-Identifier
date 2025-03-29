import numpy as np
from scipy.interpolate import interp1d
from schainpy.model import VoltageReader
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
#import plotly.graph_objects as go
#import plotly.subplots as sp
#import plotly.io as pio
import webbrowser
import tempfile
import os
import shutil
import psutil
import io
import base64
import time as timelib
import matplotlib.patches as patches

###################################################################### FUNCTIONS ########################################################################
def release_file(file_path):
    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
        try:
            if proc.info['open_files']:
                for file in proc.info['open_files']:
                    if file.path == file_path:
                        print(f"Cerrando proceso: {proc.info['name']} (PID: {proc.info['pid']})")
                        proc.terminate()  # Intenta cerrar el proceso de forma segura
                        #return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    #return False

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
                    if height_ls[possible_trail[0]] > exclude_range[0] and height_ls[possible_trail[-1]] < exclude_range[1]:
                        trail= (possible_trail[0],possible_trail[-1],j)    ### (START,END,COLUMN)
                        trails.append(trail) 
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

path = r'C:\Users\cg_h2\Documents\JRO\data'

channels=[0,1,2]
decode= False
code=[1,1,1,1,1,-1,-1,1,1,-1,1,-1,1]
nBaud=13 

startDate=dt.date(2000,6,5)
endDate=dt.date(2025,6,5)
startTime=dt.time(0,0,0)
endTime=dt.time(23,59,59)

profiles_lim= 20000 ## To avoid filling the memory when reading large files. Set 1e20 to deactivate.
exclude_range = [1e20,1e20] ## Exclude trails that start or end in this range
###################################################################### FILTER PARAMETERS ########################################################################
significance_filter_units='linear'
nSigma=5
min_channels=3
min_samples=3

###################################################################### FIGS INIT ########################################################################
figs_trails=[]

if __name__ == '__main__':
    ###################################################################### TEMP DIR PREPARATION ########################################################################
    tmp_dir_path=os.path.join(path,'temp_reader/')
    try:
        os.makedirs(tmp_dir_path, exist_ok=True)
        print(f"Temporary directory '{tmp_dir_path}' created successfully.")
    except FileExistsError:
        for file in list_files(path=tmp_dir_path,extension='.r')[1]:
            _=release_file(file)
            shutil.move(file, path)
        print(f"Temporary directory '{tmp_dir_path}' already exists.")

    ###################################################################### FILES ITERATION ########################################################################
    file_paths=list_files(path=path,extension='.r')[1]
    file_paths=sorted(file_paths)#[7:]

    code_vec=np.asarray(code)

    for file in file_paths:
        _=release_file(file)
        if os.path.exists(os.path.join(tmp_dir_path, os.path.basename(file))):  
            os.remove(os.path.join(tmp_dir_path, os.path.basename(file)))
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
                    break
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
            time_indices=np.linspace(0,len(time)-1,len(time),dtype=int)
            time_labels = [datetime.fromtimestamp(t).strftime("%H:%M") for t in data[0]]
            time_dt_full = np.array([f'{datetime.fromtimestamp(t)} | {x_i}' for t,x_i in zip(data[0],time_indices)])
            tick_idx = np.arange(0, len(data[0]), len(data[0]) // 10)
            tick_vals=time_dt_full[tick_idx]
            tick_text = [time_labels[i] for i in tick_idx]

            vmin, vmax = mat.min(), mat.max()  
            
            num_ticks_x=10
            num_ticks_y=10
            
            for trail_ID,trail in enumerate(trails):
                i_min, i_max = max(trail[2]-zoom_time,0), min(trail[2]+zoom_time,len(time_dt_full)-1)  ## Crop the RTI arround the trail
                j_min, j_max = max(trail[0]-zoom_range,0), min(trail[1]+zoom_range,len(height_ls)-1) 
                zoomed_mat=mat[j_min:j_max+1,i_min:i_max+1]

                fig, ax = plt.subplots(figsize=(15, 10))
                
                c = ax.imshow(zoomed_mat, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
                cb = plt.colorbar(c, ax=ax)
                cb.set_label("Power (dB)")
                ax.set_xlabel("Time")
                ax.set_ylabel("Range (km)")  
                ax.set_title(f"Possible event in t={time_dt_full[trail[2]]} and Range = [{height_ls[trail[0]]:.2f}, {height_ls[trail[1]]:.2f}] km")
                
                x_ticks=time_dt_full[i_min:i_max+1]         ## Ticks
                y_ticks=height_ls[j_min:j_max+1]
                tick_idx_x = np.linspace(0, len(x_ticks) - 1, num_ticks_x, dtype=int)
                tick_idx_y = np.linspace(0, len(y_ticks) - 1, num_ticks_y, dtype=int)
                ax.set_xticks(tick_idx_x)
                ax.set_xticklabels([x_ticks[i] for i in tick_idx_x], rotation=90)
                ax.set_yticks(tick_idx_y)
                ax.set_yticklabels([f'{y_ticks[i]:.2f}' for i in tick_idx_y]) 
        
                rect_x = trail[2] - i_min -1 ## Draw red box arround trail
                rect_y = trail[0] - j_min -1 
                rect_w = 2 
                rect_h = (trail[1] - trail[0])+2  
                rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h, linewidth=1.5, edgecolor='magenta', facecolor='none')
                ax.add_patch(rect)
                
                fig.tight_layout()           ## Saving plt figure for inclusion on the html dashboard
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                image_base64 = base64.b64encode(buf.read()).decode("utf-8")
                image_html = f' FILE: {os.path.basename(file)}<br> ID: {trail_ID}<br> <img src="data:image/png;base64,{image_base64}" width="1000"/>'
                
                figs_trails.append(image_html)
                plt.close(fig)


        ###################################################################### DASHBOARD DISPLAYING ########################################################################    
        timelib.sleep(1)
        _=release_file(tmp_file_path)
        if os.path.exists(os.path.join(path, os.path.basename(tmp_file_path))):  
            _=release_file(os.path.join(path, os.path.basename(tmp_file_path)))
            os.remove(os.path.join(path, os.path.basename(tmp_file_path)))
            
        shutil.move(tmp_file_path, path)
        timelib.sleep(1)  
        print('File processing completed')

    if len(figs_trails)==0:
        figs_trails=['No possible events found']

    figs_trails_str = '<br>'.join(figs_trails) 
    html_string = f"""
    <!DOCTYPE html>
    <html>
    <body>

        <h1>{len(figs_trails)} possible events found</h1>
        {figs_trails_str}
        
    </body>]
    </html>
    """

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as tmp:
        tmp.write(html_string)
        temp_file_path = tmp.name
        webbrowser.open(f"file://{temp_file_path}")
            
    os.rmdir(os.path.join(path,'temp_reader'))
