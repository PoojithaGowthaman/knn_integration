import numpy as np
import csv
import warnings

def euclidean_distance(point1,point2):
    distance=0
    try:
        for i in range(len(point1)):
            distance+=(point1[i]-point2[i])**2
        return np.sqrt(distance)
    except IndexError as e:
        print('Array lengths do not match. Points must be of equal dimensions to compute euclidean distance.')
        return np.NaN

class InvalidSubset(Exception):
    pass
class InvalidNumPredictors(Exception):
    def __init__(self,obs_len,model_len):
        self.obs_len=obs_len
        self.model_len=model_len
    def __str__(self):
        return 'Model training data contains '+str(self.model_len)+' predictors, while inputted observation contains '+str(self.obs_len)+' predictors'

def generate_prediction(knn_model,new_obs,subset): # Fix DRY violation
    if subset not in ['train','all']:
        raise InvalidSubset(str(subset)+' is invalid. Subset must be one of "train" or "all".')
    if len(new_obs)!=knn_model.x_train.shape[1]:
        raise InvalidNumPredictors(obs_len=len(new_obs),model_len=knn_model.x_train.shape[1])
    # Compute distance from new observation to every sample in training set
    if subset=='train':
        distances=np.apply_along_axis(lambda x:euclidean_distance(x,new_obs), 1, knn_model.x_train)
        # Get index positions of k closest points
        k_indices=np.argsort(distances)[:knn_model.k]
        if knn_model.model_type=='regressor':
            return np.mean(knn_model.y_train[k_indices])
        elif knn_model.model_type=='classifier':
            classes,counts=np.unique(knn_model.y_train[k_indices],return_counts=True)
            # Need to test if warning works as expected
            if len(counts)>1:
                top_counts=np.sort(counts)[-2:]
                if top_counts[0]==top_counts[1]:
                    warnings.warn('Warning: A tie has occurred (top two classes in K nearest neighbors have the same number of occurances). Classification depends on the order of the training data.')
            return np.array(classes[np.argmax(counts)],dtype='object')
        else:
            print('Invalid model type for inputted model')
            return None
    else:
        distances=np.apply_along_axis(lambda x:euclidean_distance(x,new_obs), 1, knn_model.x)
        # Get index positions of k closest points
        k_indices=np.argsort(distances)[:knn_model.k]
        if knn_model.model_type=='regressor':
            return np.mean(knn_model.y[k_indices])
        elif knn_model.model_type=='classifier':
            classes,counts=np.unique(knn_model.y[k_indices],return_counts=True)
            if len(counts)>1:
                top_counts=np.sort(counts)[-2:]
                if top_counts[0]==top_counts[1]:
                    warnings.warn('Warning: A tie has occurred (top two classes in K nearest neighbors have the same number of occurances). Classification depends on the order of the training data.')
            return np.array(classes[np.argmax(counts)],dtype='object')
        else:
            print('Invalid model type for inputted model')
            return None

def generate_predictions(knn_model,new_array,subset):
    try:
        return np.apply_along_axis(lambda x:generate_prediction(knn_model,x,subset),1,new_array)
    except InvalidNumPredictors as e:
        print('At least one inputted observation contains an invalid number of predictors. Details:')
        print(e)
