from KNN.modelling import KNN_data_collection as KNN_module
from KNN.modelling import generate_predictions as gp
from .model_metrics import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CvNotPerformed(Exception):
    pass
    
class TooManyFolds(Exception):
    def __init__(self,num_folds,num_samples):
        self.num_folds=num_folds
        self.num_samples=num_samples
    def __str__(self):
        return 'Number of folds must be less than or equal to number of samples in training set. '+str(self.num_folds)+'-fold CV requested, but only '+str(self.num_samples)+' samples exist in training set.'

class CvKNN(KNN_module.KNN):
    def __init__(self,model_type,num_folds=5):
        if model_type=='regressor' or model_type=='classifier':
            self.model_type = model_type
        else:
            raise ValueError(str(model_type)+' is not a valid model type, must be "regressor" or "classifier"')
        self.num_folds=num_folds
        print('Created CV instance for KNN '+str(model_type)+' with '+str(num_folds)+' folds!')

    def generate_cv_prediction(self,x_train,y_train,new_obs,k):
        distances=np.apply_along_axis(lambda x:gp.euclidean_distance(x,new_obs),1,x_train)
        k_indices=np.argsort(distances)[:k]
        if self.model_type=='regressor':
            return np.mean(y_train[k_indices])
        elif self.model_type=='classifier':
            classes,counts=np.unique(y_train[k_indices],return_counts=True)
            return np.array(classes[np.argmax(counts)],dtype='object')

    def generate_cv_predictions(self,x_train,y_train,x_test,k):
        return np.apply_along_axis(lambda x:self.generate_cv_prediction(x_train,y_train,x,k),1,x_test)

    def perform_cv(self,k_values):
        try:
            fold_size=len(self.x_train)//self.num_folds
            split_indices=np.arange(0,len(self.x_train)+1,fold_size)
        except ZeroDivisionError as e:
            raise TooManyFolds(num_folds=self.num_folds,num_samples=self.x_train.shape[0])
        self.__k_results=[]
        self.__k_values=[]
        for k in k_values:
            self.__k_values.append(k)
            fold_results=[]
            for i in range(self.num_folds):
                indices_to_delete=np.arange(split_indices[i],split_indices[i+1])
                x_train=np.delete(self.x_train,indices_to_delete,axis=0)
                y_train=np.delete(self.y_train,indices_to_delete,axis=0)
                x_test=self.x_train[split_indices[i]:split_indices[i+1]]
                y_test=self.y_train[split_indices[i]:split_indices[i+1]]
                fold_predictions=self.generate_cv_predictions(x_train,y_train,x_test,k)
                if self.model_type=='regressor':
                    fold_results.append(model_mse(y_test,fold_predictions))
                elif self.model_type=='classifier':
                    fold_results.append(model_misclassification(y_test,fold_predictions))
            self.__k_results.append(np.mean(fold_results))
        print('Successfully performed '+str(self.num_folds)+' fold CV!')

    def get_cv_results(self):
        try:
            # Check if CV has been performed
            self.__k_values
        except AttributeError as e:
            raise CvNotPerformed('Cross validation has not been performed. Run perform_cv function to access results')
        for i in range(len(self.__k_values)):
            print('k='+str(self.__k_values[i])+': '+str(self.__k_results[i]))
        fig,axs=plt.subplots()
        sns.lineplot(ax=axs,y=self.__k_results,x=self.__k_values)
        axs.set_title(str(self.num_folds)+' fold CV Results')
        if self.model_type=='regressor':
            axs.set_ylabel('MSE')
        elif self.model_type=='classifier':
            axs.set_ylabel('Misclassification Rate')
        axs.set_xlabel('k Value')
        plt.show()
        return axs

    def get_best_k(self):
        try:
            # Check if CV has been performed
            self.__k_values
        except AttributeError as e:
            raise CvNotPerformed('Cross validation has not been performed. Run perform_cv function to get best k')
        min_position=np.argmin(self.__k_results)
        best_k=self.__k_values[min_position]
        self.best_k=best_k
        print('Best k: '+str(best_k))
        if self.model_type=='regressor':
            print('CV MSE: '+str(self.__k_results[min_position]))
        elif self.model_type=='classifier':
            print('CV Misclassification Rate: '+str(self.__k_results[min_position]))
        self.k=best_k
