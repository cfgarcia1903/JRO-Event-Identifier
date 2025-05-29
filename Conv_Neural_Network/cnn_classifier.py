import pickle
import numpy as np
import os
import sys
from contextlib import contextmanager
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
@contextmanager
def silence_tensorflow():
    stderr = sys.stderr
    stdout = sys.stdout
    sys.stderr = open(os.devnull, 'w')
    sys.stdout = open(os.devnull, 'w')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stdout.close()
        sys.stderr = stderr
        sys.stdout = stdout

######################################################################

model_path = '/home/pc-igp-173/Documentos/neural_network/CNN_Classifier.pickle'
unclassified_path='/home/pc-igp-173/Documentos/DATA/unclassified_data/trails_JUN_2024_Day2-4.pickle'
classified_with_mistakes_path='/home/pc-igp-173/Documentos/DATA/classified_data/trails_JUN_2024_Day2-4_with_mistakes.pickle'
RTI_path='/home/pc-igp-173/Documentos/neural_network/RTIs/trails_JUN_2024_Day2-4/old'


#####################################################################
if __name__ == '__main__':
    with silence_tensorflow():
        from CV_Model import CV_Model
        with open(model_path,'rb') as f:
            CNN_classifier = pickle.load(f)

    trails = []
    with open( unclassified_path, 'rb') as f:
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
    print(f"trails: {i}")

    unclassified_matrices=[trail['Norm_Matrix'] for trail in trails]
    unclassified_matrices = np.array(unclassified_matrices)   
    unclassified_matrices = unclassified_matrices[..., np.newaxis]
    print(f'unclassified_matrices.shape: {unclassified_matrices.shape}')

    y_pred=CNN_classifier.predict_class(unclassified_matrices)
    print(f'A total of {np.sum(y_pred)} possible cosmic rays were found')

    for trail,pred_class in tqdm(zip(trails,y_pred)):
        trail['CosmicRay']=bool(pred_class)

    gc.collect()

    os.makedirs(os.path.join(RTI_path,'CR'), exist_ok=True)
    cr_path=os.path.join(RTI_path,'CR')
    os.makedirs(os.path.join(RTI_path,'not_CR'), exist_ok=True)
    not_cr_path=os.path.join(RTI_path,'not_CR')

    for i in tqdm(range(len(trails))):
        trail = trails[i]
        
        pred_class=trail['CosmicRay']
        
        fig, ax = plt.subplots()
        cax = ax.imshow(trail['Norm_Matrix'], cmap='jet', vmin=0, vmax=1, aspect='auto')
        fig.colorbar(cax)
        ax.set_title(f'event: {i}, class: {pred_class}')
        
        plot_basename=f'event{i}-{pred_class}.png'
        
        if pred_class==True:
            plot_path=os.path.join(cr_path,plot_basename)
        elif pred_class==False:
            plot_path=os.path.join(not_cr_path,plot_basename)
        else:
            print(f'Event {i} unclassified')
            continue
            
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        del fig, ax, cax, trail
        if i % 30 == 0:
            gc.collect()

    with open(classified_with_mistakes_path, 'ab') as f:
        for trail in tqdm(trails):
            pickle.dump(trail, f)
            
