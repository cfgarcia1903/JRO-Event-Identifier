import numpy as np
class CV_Model:
    def __init__(self,trained_models):
        self.models=trained_models
        self.thr=None

    def set_threshold(self,threshold):
        self.thr=threshold

    def predict_prob(self,X):
        votes_ls=[]
        for model in self.models:
            votes=model.predict(X)
            votes_ls.append(votes)
        votes_ls=np.asarray(votes_ls)
        y_pred= np.mean(votes_ls, axis=0)
        return y_pred.flatten()

    def predict_class(self,X):
        if self.thr is None:
            print('Set Threshold before calling this method')
        else:
            votes_ls=[]
            for model in self.models:
                votes=model.predict(X)
                votes_ls.append(votes)
            votes_ls=np.asarray(votes_ls)
            y_pred= np.mean(votes_ls, axis=0)
            return (y_pred.flatten()>self.thr).astype(int)


        
        