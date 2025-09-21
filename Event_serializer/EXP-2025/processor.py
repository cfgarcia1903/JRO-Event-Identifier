import matplotlib 
matplotlib.use('Agg')
from os.path import join

#######################################################
####                 MODULES                    ####
#######################################################

import parameters
import utils
import classes

#######################################################
####                 PROCESSING FUNCTION           ####
#######################################################
def process_file_hybrid(raw_file_name):
    from schainpy.model import VoltageReader
    rawdataObj = VoltageReader()
    rawdataObj.name='VoltageReader'

    times=[]
    profiles=[]
    block =1

    while not(rawdataObj.flagNoMoreFiles):    
        try:
            rawdataObj.run(path = parameters.processing_hub_path,
                        startDate=parameters.startDate,
                        endDate=parameters.endDate,
                        startTime=parameters.startTime,
                        endTime=parameters.endTime,
                        online=0,
                        walk=0,
                        expLabel='',
                        delay=5)

            profiles.append(rawdataObj.dataOut.data) 
            times.append(rawdataObj.dataOut.utctime)

            if rawdataObj.__hasNotDataInBuffer():
                ranges = rawdataObj.dataOut.heightList
                active_data,passive_data= utils.separate_data(profiles,times)

                activeRTI = classes.RTI_matrix(active_data['profiles'], ranges, active_data['times'], channels=parameters.channels, decode=parameters.decode, code_vec=parameters.code_vec, nBaud=parameters.nBaud)
                passiveRTI = classes.RTI_matrix(passive_data['profiles'], ranges, passive_data['times'], channels=parameters.channels, decode=parameters.decode, code_vec=parameters.code_vec, nBaud=parameters.nBaud)
                
                activeRTI.significance_filter(nSigma=parameters.nSigma, significance_filter_units=parameters.significance_filter_units)
                passiveRTI.significance_filter(nSigma=parameters.nSigma, significance_filter_units=parameters.significance_filter_units)

                activeRTI.coincidence_filter(min_channels=parameters.min_channels)
                passiveRTI.coincidence_filter(min_channels=parameters.min_channels)

                activeRTI.shape_filter(min_samples=parameters.min_samples)
                passiveRTI.shape_filter(min_samples=parameters.min_samples)

                active_output_path=f'{raw_file_name[:-2]}_B{block}_active.pickle'
                passive_output_path=f'{raw_file_name[:-2]}_B{block}_passive.pickle'
                active_output_path = join(parameters.output_root_path, active_output_path)
                passive_output_path = join(parameters.output_root_path, passive_output_path)

                activeRTI.process_trails(zoomed_time_size=parameters.zoomed_time_size, zoomed_range_size=parameters.zoomed_range_size, output_path_pickle=active_output_path,raw_file_name=raw_file_name)
                passiveRTI.process_trails(zoomed_time_size=parameters.zoomed_time_size, zoomed_range_size=parameters.zoomed_range_size, output_path_pickle=passive_output_path,raw_file_name=raw_file_name)

                ##### Reset data structure
                block+=1
                del activeRTI, passiveRTI
                del active_data, passive_data
                del ranges,times,profiles
                profiles = []
                times = []
                #####
                print('Next Block')
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")
            break
    del rawdataObj
    del VoltageReader