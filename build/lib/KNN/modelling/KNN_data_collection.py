import numpy as np
import csv

class KNN:
    def __init__(self,model_type,k=3):
        self.k = k
        if model_type=='regressor' or model_type=='classifier':
            self.model_type = model_type
        else:
            raise ValueError(str(model_type)+' is not a valid model type, must be "regressor" or "classifier"')
        print('Created KNN '+str(model_type)+' with k='+str(k)+'!')

    def load_csv(self,path,response):
        data_lst=[]
        with open(path ,'r') as file: # Add exception handling if file doesn't exist
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data_lst.append(row)
        # Extract column names
        column_names=data_lst[0]
        # Convert dataset to numpy array
        data_np=np.array(data_lst[1:],dtype="object")
        # Find column number of response var
        if response in column_names:
            response_position=column_names.index(response)
        else:
            raise ValueError(str(response)+' not found in dataset')
        # Split into x,y and store as instance attribute
        self.x=np.delete(data_np,response_position,axis=1).astype(float)
        if self.model_type=='regressor':
            self.y=data_np[:,response_position].astype(float)
        else:
            self.y=data_np[:,response_position]
        print('Dataset successfully loaded!')

    def train_test_split(self, test_size = 0.3):
        # Select indices for test/train split
        test_indices=np.random.choice(np.arange(0,int(len(self.y))), size = int(test_size*len(self.y)), replace = False)
        self.x_test=self.x[np.isin((np.arange(0,int(len(self.x)))),test_indices),:]
        self.y_test=self.y[np.isin((np.arange(0,int(len(self.y)))),test_indices)]
        self.x_train=self.x[~np.isin((np.arange(0,int(len(self.x)))),test_indices),:]
        self.y_train=self.y[~np.isin((np.arange(0,int(len(self.y)))),test_indices)]
        print('Successfully completed train/test split!')
        print('Training set: '+str(len(self.y_train))+' samples')
        print('Test set: '+str(len(self.y_test))+' samples')
