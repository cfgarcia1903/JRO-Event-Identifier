from numpy import where, asarray,correlate,conjugate,zeros,log10,mean,std,sum,all
from numpy import complex64,int8,float64

from os.path import basename
import parameters
import utils
import pickle

class RTI_matrix:
    def __init__(self, profiles, ranges, times, channels=parameters.channels, decode=parameters.decode, code_vec=parameters.code_vec, nBaud=parameters.nBaud, name = 'RTI'):
        ## METADATA
        self.name = name

        self.n_ranges = len(profiles[0][0])
        self.n_times = len(profiles)
        self.n_channels = len(channels)

        self.ranges = ranges
        self.times = times
        self.channels = channels

        ## DATA ARRAYS
        #print((self.n_ranges, self.n_times, self.n_channels))  FOR DEBUGGING PURPOSES
        self.voltage = zeros((self.n_ranges, self.n_times, self.n_channels), dtype=complex64)
        if decode:
            self.uncut_ranges=ranges
            self.ranges=self.ranges[:self.n_ranges-nBaud+1]
            self.n_ranges=self.n_ranges-nBaud+1
            self.voltage_decoded = zeros((self.n_ranges, self.n_times, self.n_channels), dtype=complex64)
        else:
            self.voltage_decoded = None
            self.uncut_ranges=None

        self.power_lin = zeros((self.n_ranges, self.n_times, self.n_channels), dtype=float64)

        for j, profile in enumerate(profiles):
            for k, ch in enumerate(channels):
                complex_voltage=profile[ch]
                if decode:
                    complex_voltage_decoded=utils.decode_signal(complex_voltage,code_vec)
                    self.voltage_decoded[:, j, k] = asarray(complex_voltage_decoded) 
                    power = (conjugate(complex_voltage_decoded)*complex_voltage_decoded).real
                else:
                    power = (conjugate(complex_voltage)*complex_voltage).real

                self.voltage[:, j, k] = asarray(complex_voltage)
                self.power_lin[:, j, k] = asarray(power)  

        self.power_db = 10 * log10(self.power_lin + 1)  

    def significance_filter(self, nSigma=parameters.nSigma, significance_filter_units=parameters.significance_filter_units):
        self.significance_mask = zeros((self.n_ranges, self.n_times, self.n_channels), dtype=bool)

        if significance_filter_units=='linear':
            power_matrices=self.power_lin
        elif significance_filter_units=='dB':
            power_matrices=self.power_db
        else:
            self.significance_mask = None
            raise ValueError("significance_filter_units must be 'linear' or 'dB'")

        for k in range(self.n_channels):
            for i in range(self.n_ranges):
                pow_vec=power_matrices[i,:,k]
                significance_threshold=mean(pow_vec)+nSigma*std(pow_vec)
                self.significance_mask[i,:,k]= pow_vec>=significance_threshold

    def coincidence_filter(self, min_channels=parameters.min_channels):
        self.coincidence_mask = (sum(self.significance_mask.astype(int8), axis=2) >= min_channels).astype(float64)
        
        self.power_lin_joint = sum(self.power_lin, axis=2)
        self.power_db_joint = 10 * log10(self.power_lin_joint + 1)

    def save_block(self, output_path_pickle=None,raw_file_name=None):
        block_data={
                #CONTEXT DATA
                'file':basename(raw_file_name),
                'ranges':self.ranges,'uncut_ranges':self.uncut_ranges,
                'vmin':min(self.power_db_joint), 'vmax':max(self.power_db_joint),
                'times':self.times,
                #ARRAYS
                'dB_joint':self.power_db_joint, 'lin_joint':self.power_lin_joint,
                'db_ch':self.power_db, 'lin_ch':self.power_lin,
                'voltage':self.voltage, 'voltage_decoded': None,
                'coincidence_mask':self.coincidence_mask
                }
        
        if self.voltage_decoded is not None:
            block_data['voltage_decoded'] = self.voltage_decoded

        with open(output_path_pickle, 'ab') as f:
            pickle.dump(block_data, f)
            print(f'block stored in {output_path_pickle}')
        
       
