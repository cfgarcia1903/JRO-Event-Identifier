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

        #print(f'Significance filter completed ({self.name})', end=' | ')

    def coincidence_filter(self, min_channels=parameters.min_channels):
        self.coincidence_mask = (sum(self.significance_mask.astype(int8), axis=2) >= min_channels).astype(float64)
        
        self.power_lin_joint = sum(self.power_lin, axis=2)
        self.power_db_joint = 10 * log10(self.power_lin_joint + 1)

        self.power_lin_joint_significant = self.coincidence_mask * self.power_lin_joint
        self.power_db_joint_significant = self.coincidence_mask * self.power_db_joint

        #print('coincidence filter completed ({self.name})', end=' | ')

    def shape_filter(self, min_samples=parameters.min_samples):

        up_lim = self.n_ranges - 1
        lo_lim = 0
        trails = []
        
        center_columns = self.coincidence_mask[:, 1:-1]  # j columns (middle columns)

        candidate_counts = sum(center_columns > 0, axis=0)
        valid_time_indices = where(candidate_counts >= min_samples)[0] + 1  # +1 to adjust for slicing offset

        for j in valid_time_indices:
            col_vec=self.coincidence_mask[:,j]
            candidate_rows=where(col_vec > 0)[0]
            possible_trails = utils.find_sequences(candidate_rows, min_size=min_samples)
            
            if not possible_trails:
                continue
    
            for possible_trail in possible_trails:
                includes_lo= possible_trail[0]==lo_lim
                includes_up= possible_trail[-1]==up_lim
                
                if includes_lo and includes_up:
                    side= list(possible_trail)
                elif includes_lo:
                    side=list(possible_trail)+[int(possible_trail[-1]+1)]
                elif includes_up:
                    side=[int(possible_trail[0]-1)]+list(possible_trail)
                else:
                    side=[int(possible_trail[0]-1)]+list(possible_trail)+[int(possible_trail[-1]+1)]

                left_clear= all(self.coincidence_mask[side,j-1]==0)
                right_clear= all(self.coincidence_mask[side,j+1]==0)

                if left_clear and right_clear:
                    trail= (possible_trail[0],possible_trail[-1],j)    ### (START,END,COLUMN)
                    trails.append(trail) 

        self.trails=trails
        #print('shape filter completed ({self.name})', end=' | ')

    def process_trails(self, zoomed_time_size=parameters.zoomed_time_size, zoomed_range_size=parameters.zoomed_range_size,
                        output_path_pickle=None,raw_file_name=None):
        print(f'({self.name}) Number of trails found: {len(self.trails)}', end=' | ')
        stored_crs=0
        if len(self.trails):

            vmin, vmax = self.power_db_joint.min(), self.power_db_joint.max()

            for trail_ID,trail in enumerate(self.trails):
                
                j_size=zoomed_time_size
                i_size=zoomed_range_size

                j_mid,i_mid= trail[2],int((trail[0]+trail[1])/2)

                if j_mid-j_size//2 < 0:
                    j_min=0
                    j_max=j_min+j_size
                elif j_mid+j_size//2 > self.n_times-1:
                    j_max=self.n_times-1
                    j_min=j_max-j_size
                else:
                    j_min=j_mid-j_size//2
                    j_max=j_min+j_size

                if i_mid-i_size//2 < 0:
                    i_min=0
                    i_max=i_min+i_size
                elif i_mid+i_size//2 > self.n_ranges-1:
                    i_max=self.n_ranges-1
                    i_min=i_max-i_size
                else:
                    i_min=i_mid-i_size//2
                    i_max=i_min+i_size

                zoomed_db_joint=self.power_db_joint[i_min:i_max,j_min:j_max]
                zoomed_lin_joint=self.power_lin_joint[i_min:i_max,j_min:j_max]
                zoomed_voltage=self.voltage[i_min:i_max,j_min:j_max,:]
                voltage_profile=self.voltage[:,trail[2],:]
                trail_power_db_joint=self.power_db_joint[trail[0]:trail[1]+1,trail[2]]
                trail_power_lin_joint=self.power_lin_joint[trail[0]:trail[1]+1,trail[2]]
                trail_voltage=self.voltage[trail[0]:trail[1]+1,trail[2],:]
                trail_power_db=self.power_db[trail[0]:trail[1]+1,trail[2],:]
                trail_power_lin=self.power_lin[trail[0]:trail[1]+1,trail[2],:]
                if self.voltage_decoded is not None:
                    zoomed_voltage_decoded=self.voltage_decoded[i_min:i_max,j_min:j_max,:]
                    voltage_decoded_profile=self.voltage_decoded[:,trail[2],:]
                    trail_voltage_decoded=self.voltage_decoded[trail[0]:trail[1]+1,trail[2],:]


                trail_data={
                        #CONTEXT DATA
                        'ID':trail_ID,'CosmicRay': None,'file':basename(raw_file_name),
                        'ranges':self.ranges,'uncut_ranges':self.uncut_ranges,
                        'vmin':vmin, 'vmax':vmax,
                        #TRAIL LOCATION
                        'time_ID':trail[2],'timestamp':self.times[trail[2]],
                        'range_start_ID':trail[0],'range_end_ID':trail[1],'range_start':self.ranges[trail[0]],'range_end':self.ranges[trail[1]],
                        'zoomed_time':self.times[j_min:j_max],'zoomed_range':self.ranges[i_min:i_max],
                        #TRAIL ARRAYS
                        'trail_power_dB_joint':trail_power_db_joint,'trail_power_lin_joint':trail_power_lin_joint,
                        'trail_power_dB':trail_power_db,'trail_power_lin':trail_power_lin,
                        'trail_voltage':trail_voltage, 'trail_voltage_decoded': None,
                        #ZOOMED ARRAYS
                        'zoomed_dB_joint':zoomed_db_joint, 'zoomed_lin_joint':zoomed_lin_joint,
                        'zoomed_voltage':zoomed_voltage, 'zoomed_voltage_decoded': None,
                        'zoomed_dB_joint_normalized': (zoomed_db_joint-vmin)/(vmax-vmin),
                        'coincidence_mask_zoomed':self.coincidence_mask[i_min:i_max,j_min:j_max],
                        #PROFILE
                        'voltage_profile':voltage_profile, 'voltage_decoded_profile': None
                        }
                
                if self.voltage_decoded is not None:
                    trail_data['trail_voltage_decoded']=trail_voltage_decoded
                    trail_data['zoomed_voltage_decoded']=zoomed_voltage_decoded
                    trail_data['voltage_decoded_profile']=voltage_decoded_profile
                
                with open(output_path_pickle, 'ab') as f:
                    pickle.dump(trail_data, f)
                stored_crs += 1
            print(f'{stored_crs} trails were stored in {output_path_pickle}')
        else:
            print('No trails to process')
       
    def process_trails_var(self, zoomed_time_size=parameters.zoomed_time_size, zoomed_range_size=parameters.zoomed_range_size,
                        output_path_pickle=None,raw_file_name=None):
        print(f'({self.name}) Number of trails found: {len(self.trails)}', end=' | ')
        stored_crs=0
        if len(self.trails):

            vmin, vmax = self.power_db_joint.min(), self.power_db_joint.max()

            file_summary={
                'file':basename(raw_file_name),
                'coincidence_mask':self.coincidence_mask,
                'power_db_joint':self.power_db_joint,
                'power_lin_joint':self.power_lin_joint,
                'ranges':self.ranges,
                'times':self.times,
                'vmin':vmin,
                'vmax':vmax
                }
            with open(output_path_pickle.split('.')[0] + '.sum', 'wb') as f:
                pickle.dump(file_summary, f)

            for trail_ID,trail in enumerate(self.trails):
                
                j_size=zoomed_time_size
                i_size=zoomed_range_size

                j_mid,i_mid= trail[2],int((trail[0]+trail[1])/2)

                if j_mid-j_size//2 < 0:
                    j_min=0
                    j_max=j_min+j_size
                elif j_mid+j_size//2 > self.n_times-1:
                    j_max=self.n_times-1
                    j_min=j_max-j_size
                else:
                    j_min=j_mid-j_size//2
                    j_max=j_min+j_size

                if i_mid-i_size//2 < 0:
                    i_min=0
                    i_max=i_min+i_size
                elif i_mid+i_size//2 > self.n_ranges-1:
                    i_max=self.n_ranges-1
                    i_min=i_max-i_size
                else:
                    i_min=i_mid-i_size//2
                    i_max=i_min+i_size

                zoomed_db_joint=self.power_db_joint[i_min:i_max,j_min:j_max]
                zoomed_lin_joint=self.power_lin_joint[i_min:i_max,j_min:j_max]
                zoomed_voltage=self.voltage[i_min:i_max,j_min:j_max,:]
                voltage_profile=self.voltage[:,trail[2],:]
                trail_power_db_joint=self.power_db_joint[trail[0]:trail[1]+1,trail[2]]
                trail_power_lin_joint=self.power_lin_joint[trail[0]:trail[1]+1,trail[2]]
                trail_voltage=self.voltage[trail[0]:trail[1]+1,trail[2],:]
                trail_power_db=self.power_db[trail[0]:trail[1]+1,trail[2],:]
                trail_power_lin=self.power_lin[trail[0]:trail[1]+1,trail[2],:]
                if self.voltage_decoded is not None:
                    zoomed_voltage_decoded=self.voltage_decoded[i_min:i_max,j_min:j_max,:]
                    voltage_decoded_profile=self.voltage_decoded[:,trail[2],:]
                    trail_voltage_decoded=self.voltage_decoded[trail[0]:trail[1]+1,trail[2],:]


                trail_data={
                        #CONTEXT DATA
                        'ID':trail_ID,'CosmicRay': None,'file':basename(raw_file_name),
                        'ranges':self.ranges,'uncut_ranges':self.uncut_ranges,
                        'vmin':vmin, 'vmax':vmax,
                        #TRAIL LOCATION
                        'time_ID':trail[2],'timestamp':self.times[trail[2]],
                        'range_start_ID':trail[0],'range_end_ID':trail[1],'range_start':self.ranges[trail[0]],'range_end':self.ranges[trail[1]],
                        'zoomed_time':self.times[j_min:j_max],'zoomed_range':self.ranges[i_min:i_max],
                        #TRAIL ARRAYS
                        'trail_power_dB_joint':trail_power_db_joint,'trail_power_lin_joint':trail_power_lin_joint,
                        'trail_power_dB':trail_power_db,'trail_power_lin':trail_power_lin,
                        'trail_voltage':trail_voltage, 'trail_voltage_decoded': None,
                        #ZOOMED ARRAYS
                        'zoomed_dB_joint':zoomed_db_joint, 'zoomed_lin_joint':zoomed_lin_joint,
                        'zoomed_voltage':zoomed_voltage, 'zoomed_voltage_decoded': None,
                        'zoomed_dB_joint_normalized': (zoomed_db_joint-vmin)/(vmax-vmin),
                        'coincidence_mask_zoomed':self.coincidence_mask[i_min:i_max,j_min:j_max],
                        #PROFILE
                        'voltage_profile':voltage_profile, 'voltage_decoded_profile': None
                        }
                
                if self.voltage_decoded is not None:
                    trail_data['trail_voltage_decoded']=trail_voltage_decoded
                    trail_data['zoomed_voltage_decoded']=zoomed_voltage_decoded
                    trail_data['voltage_decoded_profile']=voltage_decoded_profile
                
                with open(output_path_pickle, 'ab') as f:
                    pickle.dump(trail_data, f)
                stored_crs += 1
            print(f'{stored_crs} trails were stored in {output_path_pickle}')
        else:
            print('No trails to process')
       














