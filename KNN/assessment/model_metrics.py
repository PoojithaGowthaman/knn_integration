from ..modelling import KNN_data_collection as KNN_module
from ..modelling import generate_predictions as gp
import numpy as np

## Metrics for classification models
def model_accuracy(actual,predicted): # for classifer
    if len(actual)!=len(predicted):
        raise ValueError('Length mismatch: length of actual and predicted arrays differ')
    accuracy=np.mean(np.array(predicted)==np.array(actual))
    return accuracy

def model_misclassification(actual,predicted): #for classifer
    if len(actual)!=len(predicted):
        raise ValueError('Length mismatch: length of actual and predicted arrays differ')
    misclassificaiton_rate=np.mean(np.array(predicted)!=np.array(actual))
    return misclassificaiton_rate

def model_num_correct(actual,predicted):
    if len(actual)!=len(predicted):
        raise ValueError('Length mismatch: length of actual and predicted arrays differ')
    num_correct=np.sum(np.array(actual)==np.array(predicted))
    return num_correct

def model_num_incorrect(actual,predicted):
    if len(actual)!=len(predicted):
        raise ValueError('Length mismatch: length of actual and predicted arrays differ')
    num_incorrect=np.sum(np.array(actual)!=np.array(predicted))
    return num_incorrect

## Metrics for regression models
def model_rmse(actual,predicted): # for regressor
    if len(actual)!=len(predicted):
        raise ValueError('Length mismatch: length of actual and predicted arrays differ')
    rmse=np.sqrt(np.mean((np.array(actual)-np.array(predicted))**2))
    return rmse

def model_mse(actual,predicted): # for regressor
    if len(actual)!=len(predicted):
        raise ValueError('Length mismatch: length of actual and predicted arrays differ')
    mse=np.mean((np.array(actual)-np.array(predicted))**2)
    return mse

def model_mae(actual,predicted): # for regressor
    if len(actual)!=len(predicted):
        raise ValueError('Length mismatch: length of actual and predicted arrays differ')
    mae=np.mean(np.abs(np.array(actual)-np.array(predicted)))
    return mae

def model_mape(actual,predicted):
    if len(actual)!=len(predicted):
        raise ValueError('Length mismatch: length of actual and predicted arrays differ')
    mape=(1/len(actual))*np.sum((np.abs(np.array(actual)-np.array(predicted)))/np.array(actual))*100
    return mape

def assessment_metrics(knn_model):
    test_predictions=gp.generate_predictions(knn_model,knn_model.x_test,'train')
    if knn_model.model_type =='regressor':
        print('Test MSE: '+str(model_mse(knn_model.y_test,test_predictions)))
        print('Test RMSE: '+str(model_rmse(knn_model.y_test,test_predictions)))
        print('Test MAE: '+str(model_mae(knn_model.y_test,test_predictions)))
        print('Test MAPE: '+str(model_mape(knn_model.y_test,test_predictions))+'%')
    elif knn_model.model_type =='classifier':
        print('Test accuracy: '+str(model_accuracy(knn_model.y_test,test_predictions)))
        print('Test misclassification rate: '+str(model_misclassification(knn_model.y_test,test_predictions)))
        print('Num correct predictions: '+str(model_num_correct(knn_model.y_test,test_predictions)))
        print('Num incorrect predictions: '+str(model_num_incorrect(knn_model.y_test,test_predictions)))
