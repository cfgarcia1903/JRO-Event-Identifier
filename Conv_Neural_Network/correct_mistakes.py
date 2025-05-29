import pickle
import os
import sys
from contextlib import contextmanager
from tqdm import tqdm
import chocopearl as ch






RTI_path='/home/pc-igp-173/Documentos/neural_network/RTIs/trails_JUN_2024_Day3/old'

classified_with_mistakes_path='/home/pc-igp-173/Documentos/DATA/classified_data/trails_JUN_2024_Day3_with_mistakes.pickle'
classified_path =  '/home/pc-igp-173/Documentos/DATA/classified_data/trails_JUN_2024_Day3.pickle'
cosmic_rays_path=  '/home/pc-igp-173/Documentos/DATA/cosmic_rays/trails_JUN_2024_Day3_CR.pickle'
#####################################################################
if __name__ == '__main__':

    trails = []
    with open( classified_with_mistakes_path, 'rb') as f:
        i=0
        while True:
            try:
                obj = pickle.load(f)
                trails.append(obj)
                #clear_output(wait=True)
                #print(f"trail: {i}")
                i+=1
            except EOFError as e:
                print(e)
                break
    print(f"total trails: {i}")

    print('Correcting mistakes...')
    mistakes=ch.list_files(os.path.join(RTI_path,'mistakes'))[0]
    mistake_count=0
    for mistake in tqdm(mistakes):
        info=mistake.split('-')
        id_event,former_class=info[0][5:],info[1].split('.')[0]
        id_event = int(id_event)
        new_class=not eval(former_class)
        trails[id_event]['CosmicRay'] = new_class
        mistake_count+=1
        
    print(f'{mistake_count} corrected mistakes')

    with open(classified_path, 'ab') as f:
        for trail in tqdm(trails):
            pickle.dump(trail, f)

    print('Saving only cosmic rays...')
    trails = []
    with open( classified_path, 'rb') as f:
        i=0
        while True:
            try:
                obj = pickle.load(f)
                trails.append(obj)
                #clear_output(wait=True)
                #print(f"trail: {i}")
                i+=1
            except EOFError as e:
                print(e)
                break
        
    crs=0
    with open(cosmic_rays_path, 'ab') as f:
        for trail in tqdm(trails):
            pred_class=trail['CosmicRay']
            if pred_class==True:
                pickle.dump(trail, f)
                crs+=1
            elif pred_class==False:
                pass
            else:
                print(f'Event {i} unclassified')
    print(f'total: {crs} cosmic rays from {i} trails')
































    