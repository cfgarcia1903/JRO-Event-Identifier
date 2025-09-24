from os import getcwd,listdir
from os.path import join as pathjoin
from numpy import array, where, asarray,correlate,zeros
from scipy.interpolate import interp1d

import classes
import parameters
#######################################################
####                 OS UTILS                      ####
#######################################################

def list_files(path=None,extension=None):
    if path is None:
        path=getcwd()
    filenames_list = []
    filepaths_list = []
    if extension is None:
        for name in listdir(path):
            full_path = pathjoin(path, name)
            filenames_list.append(name)
            filepaths_list.append(full_path)
    else:
        for name in listdir(path):
            if name.endswith(extension):
                full_path = pathjoin(path, name)
                filenames_list.append(name)
                filepaths_list.append(full_path)
    return filenames_list,filepaths_list


#######################################################
####                 TIMESTAMP UTILS               ####
#######################################################

def detect_timestamp_changes(timestamps, ids):
    timestamps = array(timestamps)
    ids = array(ids)
    changes = where(timestamps[1:] != timestamps[:-1])[0] + 1
    return (ids[0], *ids[changes])

def interpolate_timestamps(timestamps_change, ids_change, full_ids):
    timestamps_change = array(timestamps_change)
    ids_change = array(ids_change)
    
    interpolator = interp1d(ids_change, timestamps_change, kind='linear', fill_value="extrapolate")

    full_ids = asarray(full_ids)

    full_timestamps = interpolator(full_ids)
    
    # Extrapolate using the last interval's slope
    if len(ids_change) > 1:
        last_slope = (timestamps_change[-1] - timestamps_change[-2]) / (ids_change[-1] - ids_change[-2])
        last_idx = ids_change[-1]
        mask = full_ids >= last_idx
        full_timestamps[mask] = timestamps_change[-1] + last_slope * (full_ids[mask] - last_idx)
    
    return full_timestamps

#######################################################
####                 DATA PROCESSING UTILS         ####
#######################################################

def separate_data(datablock,basetime,ippSeconds,cut=-20,first_passive=False):
    active_data = {}
    passive_data = {}
    data=zeros((datablock.shape[1],datablock.shape[0],datablock.shape[2]))
    for i in range(datablock.shape[1]):
        profile=datablock[:,i,:]
        data[i,:,:]=profile
    
    active_data['profiles'] = data[:cut,:,:]
    passive_data['profiles'] = data[cut:,:,:]

    times=basetime + (array(range(datablock.shape[1])) * ippSeconds)
    active_data['times'] = times[:cut]
    passive_data['times'] = times[cut:]

    if first_passive:
        active_data, passive_data = passive_data, active_data
    return active_data, passive_data


def find_sequences(arr, min_size=3):
    groups = []
    start = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1] + 1:   # detecta corte
            if i - start >= min_size:
                groups.append(arr[start:i])  # solo corta si el grupo es válido
            start = i
    if len(arr) - start >= min_size:
        groups.append(arr[start:])
    return groups


def decode(signal,code,mode='valid'):
    if mode=='valid':
        signal = asarray(signal)
        code = asarray(code)

        if len(signal) < len(code):
            raise ValueError("Code cant be larger than signal")

        decoded = correlate(signal, code, mode="valid")
        return decoded

