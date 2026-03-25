import matplotlib 
matplotlib.use('Agg')

from os.path import basename
from os import makedirs,remove
from time import sleep
import shutil

BUFFER_SIZE = 4*1024*1024
_original_copyfileobj = shutil.copyfileobj
def custom_copyfileobj(fsrc, fdst, length=BUFFER_SIZE):
    return _original_copyfileobj(fsrc,fdst,length)
shutil.copyfileobj = custom_copyfileobj

#######################################################
####                 MODULES                       ####
#######################################################

import utils
import processor

#######################################################
####                 PARAMETERS                    ####
#######################################################

from parameters import processing_hub_path, raw_files_root_path, raw_files

if __name__ == '__main__':
    ###################################################################### TEMP DIR PREPARATION ########################################################################
    try:
        makedirs(processing_hub_path)
        print(f"Temporary directory '{processing_hub_path}' created successfully.")
    except FileExistsError:
        for file in utils.list_files(path=processing_hub_path,extension='.r')[1]:
            remove(file)
        print(f"Temporary directory '{processing_hub_path}' already exists.")

    ###################################################################### LISTING RAW FILES ########################################################################
    file_paths=utils.list_files(path=raw_files_root_path,extension='.r')[1]
    file_paths=sorted(file_paths)

    ###################################################################### FILES ITERATION ########################################################################
    for file in file_paths:
        if basename(file) not in raw_files:
            continue
        tmp_file_path = shutil.copy(file, processing_hub_path)
        sleep(1)
        print(f'Copied {basename(file)} to {processing_hub_path}')

        returncode= processor.process_file_hybrid(basename(file))
        if returncode == 0:
            print(f'Processor returned code 0 while working on file {basename(file)}')
            break
        print(f'Processed {basename(file)}')

        remove(tmp_file_path)
        print(f'Removed {basename(tmp_file_path)} from {processing_hub_path}')