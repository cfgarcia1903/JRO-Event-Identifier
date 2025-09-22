import matplotlib 
matplotlib.use('Agg')
from os.path import join
from numpy import any
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
    print(f'[Processor] Processing file: {raw_file_name}')
    from schainpy.model import VoltageReader
    rawdataObj = VoltageReader()
    rawdataObj.name='VoltageReader'
    rawdataObj.setup(path = r'/home/francisco/Documentos/JRO/Release_2.0.0 test/processing_hub',
                    startDate=parameters.startDate,
                    endDate=parameters.endDate,
                    startTime=parameters.startTime,
                    endTime=parameters.endTime,
                    online=0,
                    walk=0,
                    expLabel='',
                    delay=5,
                    name='VoltageReader')
    
    block =1
    while(not rawdataObj.flagNoMoreFiles):
        try:
            datablock = rawdataObj.datablock
            basetime = rawdataObj.dataOut.utctime = rawdataObj.basicHeaderObj.utc + rawdataObj.basicHeaderObj.miliSecond/1000.
            ippSeconds = rawdataObj.radarControllerHeaderObj.ippSeconds
            ranges = rawdataObj.dataOut.heightList

            print('[Processor] Processing block No. {block}')
            active_data,passive_data= utils.separate_data(datablock,basetime,ippSeconds)

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

            ##### Memory Management
            del datablock, ranges, basetime, ippSeconds
            del activeRTI, passiveRTI
            del active_data, passive_data
            del active_output_path, passive_output_path

            ##### Next Block
            if rawdataObj.flagNoMoreFiles:
                break
            else:
                block+=1
                print('Next Block')
                rawdataObj.readNextBlock()
    
        except Exception as e:
            print(f"An error occurred while processing: {e}")
            return 0

    del rawdataObj
    del VoltageReader
    return 1