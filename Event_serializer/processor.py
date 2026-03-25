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
    rawdataObj.setup(path = parameters.processing_hub_path,
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
    ranges = None
    while(not rawdataObj.flagNoMoreFiles):
        rawdataObj.run()
        try:
            if ranges is None:
                ranges = rawdataObj.dataOut.heightList
            datablock = rawdataObj.datablock
            basetime = rawdataObj.dataOut.utctime = rawdataObj.basicHeaderObj.utc + rawdataObj.basicHeaderObj.miliSecond/1000.
            ippSeconds = rawdataObj.radarControllerHeaderObj.ippSeconds

            print(f'[Processor] Processing block No. {block}')
            if datablock is None:
                block+=1
                #print('Next Block')
                rawdataObj.readNextBlock()
                continue
            #else:    # FOR DEBUGGING PURPOSES
                #print(datablock.shape)
                #print(basetime)
                #print(ippSeconds)
                #print(len(ranges))

            active_data,passive_data= utils.separate_data(datablock,basetime,ippSeconds,cut=parameters.cut,first_passive=False)
            activeRTI = classes.RTI_matrix(active_data['profiles'], ranges, active_data['times'], channels=parameters.channels, decode=parameters.decode, code_vec=parameters.code_vec, nBaud=parameters.nBaud, name='Active')
            passiveRTI = classes.Passive_matrix(passive_data['profiles'], ranges, passive_data['times'], channels=parameters.channels, name='Passive')
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
            passiveRTI.process_trails(output_path_pickle=passive_output_path,raw_file_name=raw_file_name)
            print('') 
            print(f'{raw_file_name[:-2]}_B{block} processed')
                  

            ##### Memory Management
            del datablock, basetime, ippSeconds
            del activeRTI, passiveRTI
            del active_data, passive_data
            del active_output_path, passive_output_path

            ##### Next Block
            if rawdataObj.flagNoMoreFiles:
                break
            else:
                block+=1
                #print('Next Block')
                rawdataObj.readNextBlock()
                continue
            
        except Exception as e:
            if str(e) == 'No more files to read':
                break
            print(f"An error occurred while processing: {e}")
            return 0

    del rawdataObj
    del VoltageReader
    return 1

if __name__ == '__main__':
    process_file_hybrid('test')