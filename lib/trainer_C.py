import numpy as np
from rlxutils import subplots
import matplotlib.pyplot as plt

TRAIN, TEST, VAL = 0,1,2

class Trainer_C:
    
    def __init__(self, input_matrix_train, input_matrix_test, 
                       output_matrix_train, output_matrix_test, 
                       split_mask,
                       input_features, feature_to_predict,
                       comp_to_predict="real"):
        
        self.input_matrix_train = input_matrix_train
        self.input_matrix_test = input_matrix_test
        self.output_matrix_train = output_matrix_train
        self.output_matrix_test = output_matrix_test
        self.split_mask = split_mask
        self.input_features = input_features
        self.feature_to_predict = feature_to_predict
        self.comp_to_predict = str(comp_to_predict).lower()

        if not (input_matrix_train.shape[:2] == input_matrix_test.shape[:2] == output_matrix_train.shape[:2] == output_matrix_test.shape[:2]):
            raise ValueError("all matrices and normed coherences must have the same pixel size")

        if not (input_matrix_train.shape[2:] == input_matrix_test.shape[2:]):
            raise ValueError(f"input matrices must have the same shape {input_matrix_train.shape[2:]} vs {input_matrix_test.shape[2:]}")

        if not (output_matrix_train.shape[2:] == output_matrix_test.shape[2:]):
            raise ValueError(f"output matrices must have the same shape {output_matrix_train.shape[2:]} vs {output_matrix_test.shape[2:]}")
        
        if not self.comp_to_predict in ["real", "imag"]:
            raise ValueError("component to predict must be 'real' or 'imag'")
        
    def split(self):
        # build x
        xtr = []
        xts = []
        xnames = []
        for f in self.input_features:
            i,j = f
            
            if i == j:
                xtr.append(self.input_matrix_train[:,:,i,j][self.split_mask==TRAIN].real.flatten())
                xts.append(self.input_matrix_test[:,:,i,j][self.split_mask==TEST].real.flatten())
                xnames.append(f"{f}")
            else:
                xtr.append(self.input_matrix_train[:,:,i,j][self.split_mask==TRAIN].real.flatten())
                xtr.append(self.input_matrix_train[:,:,i,j][self.split_mask==TRAIN].imag.flatten())
                xnames.append(f"{f}.real")
                xts.append(self.input_matrix_test[:,:,i,j][self.split_mask==TEST].real.flatten())
                xts.append(self.input_matrix_test[:,:,i,j][self.split_mask==TEST].imag.flatten())
                xnames.append(f"{f}.imag")

        self.xtr = np.r_[xtr].T
        self.xts = np.r_[xts].T
        self.xnames = xnames

        # build y
        i,j = self.feature_to_predict
        if i==j or self.comp_to_predict == "real":
            self.ytr = self.output_matrix_train[:,:,i,j].real[self.split_mask==TRAIN]
            self.yts = self.output_matrix_test[:,:,i,j].real[self.split_mask==TEST]
        else:
            self.ytr = self.output_matrix_train[:,:,i,j].imag[self.split_mask==TRAIN]
            self.yts = self.output_matrix_test[:,:,i,j].imag[self.split_mask==TEST]
        return self
    
    def plot_distributions(self):
        #n = len(self.input_features)+1
        n = self.xtr.shape[-1] + 1
        for ax,i in subplots(n, usizex=4):
            
            if i < n-1:
                xtri = self.xtr[:,i]
                a,b = np.percentile(xtri, [1,99])
                xtri = xtri[(xtri>a)&(xtri<b)]
                
                xtsi = self.xts[:,i]
                a,b = np.percentile(xtsi, [1,99])
                xtsi = xtsi[(xtsi>a)&(xtsi<b)]
                
                plt.hist(xtri, bins=100, alpha=.5, density=True, label='train')
                plt.hist(xtsi, bins=100, alpha=.5, density=True, label='test')
                plt.grid()
                plt.title(f"distribution of input {self.xnames[i]}")
                plt.legend();                
            else:
                plt.hist(self.ytr, bins=100, alpha=.5, density=True, label='train')
                plt.hist(self.yts, bins=100, alpha=.5, density=True, label='test')
                plt.grid()
                plt.title(f"distribution of predictive target {self.feature_to_predict}.{self.comp_to_predict}")
                plt.legend();
                
    def set_estimator(self, estimator):
        self.estimator = estimator
        return self
    
    def fit(self):
        self.estimator.fit(self.xtr, self.ytr)
        
        self.predstr = self.estimator.predict(self.xtr)
        self.predsts = self.estimator.predict(self.xts)

        # we use mean absolute error
        self.errtr = np.mean(np.abs(self.predstr - self.ytr))
        self.errts = np.mean(np.abs(self.predsts - self.yts))
        return self

        
    def plot_predictions(self):
        for ax,i in subplots(2, usizex=5, usizey=5):
            if i==0: plt.scatter(self.ytr, self.predstr, alpha=.1, s=10); plt.title(f"train mae {self.errtr:.3f}") 
            if i==1: plt.scatter(self.yts, self.predsts, alpha=.1, s=10); plt.title(f"test mae {self.errts:.3f}")
            plt.grid()