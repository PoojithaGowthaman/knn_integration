import unittest
from KNN.modelling import KNN_data_collection as knn
from KNN.assessment import model_metrics as mm
from KNN.modelling import generate_predictions as gp
import numpy as np
import random
import numbers

#user defined exceptions
class KnnValueError(Exception):
    pass

class InvalidModel(Exception):
    pass

class TestRegressionMetrics(unittest.TestCase):
    mpg_to_predict, assert_rmse,assert_mse,assert_mae,assert_mape =0,0,0, 0, 0
    test_knn_regressor, actual_mpg,predicted_mpg,k = 0,0,0,0
    model_type = " "
    
    @classmethod
    def setUpClass(cls):
        try:
            TestRegressionMetrics.test_knn_regressor = knn.KNN("regressor",7)
            if TestRegressionMetrics.k < 0 and TestRegressionMetrics.k > 10:
                raise KnnValueError()
            if TestRegressionMetrics.model_type!= "regressor" or TestRegressionMetrics.model_type!="classifier":
                raise InvalidModel()
        except InvalidModel:
            print('K nearest neighbors is advisable to a maximum of 10 for statistical effiency')
        
        try:
            TestRegressionMetrics.test_knn_regressor.load_csv('datasets/auto_mpg.csv','mpg')
        except FileNotFoundError:
            print('Dataset not found, check the datasets folder')
            
        try:
            TestRegressionMetrics.test_knn_regressor.train_test_split(0.25)
        except ValueError:
            print('Enter a value between 0.1 to 0.9 as test-train split percentage')
        
        print('from setUpClass')
    
    def setUp(self):
        try:
            self.actual_mpg=[19,18,23,28]
            self.mpg_to_predict=np.array([[6,250,100,3282,15],[6,250,88,3139,14.5],[4,122,86,2220,14],[4,116,90,2123,14]])
        except IndexError:
            print('Ensure array lengths of actual_mpg and mpg_to_predict match')
            
        self.predicted_mpg= gp.generate_predictions(TestRegressionMetrics.test_knn_regressor,self.mpg_to_predict,'train')
        
    def test_model_rmse(self):
        TestRegressionMetrics.assert_rmse = mm.model_rmse(self.actual_mpg,self.predicted_mpg)
        self.assertEqual(len(self.actual_mpg), len(self.predicted_mpg))
        self.assertIsInstance(TestRegressionMetrics.assert_rmse, numbers.Number)
        self.assertIsInstance(TestRegressionMetrics.assert_rmse, float)
        assert TestRegressionMetrics.assert_rmse >= 0
        
    def test_model_mae(self):
        
        TestRegressionMetrics.assert_mae = mm.model_mae(self.actual_mpg,self.predicted_mpg)
        self.assertEqual(len(self.actual_mpg), len(self.predicted_mpg))
        self.assertIsInstance(TestRegressionMetrics.assert_mae, numbers.Number)
        self.assertIsInstance(TestRegressionMetrics.assert_mae, float)
        assert TestRegressionMetrics.assert_mae >= 0
        
    def test_model_mape(self):
        
        TestRegressionMetrics.assert_mape = mm.model_mape(self.actual_mpg,self.predicted_mpg)
        self.assertEqual(len(self.actual_mpg), len(self.predicted_mpg))
        self.assertIsInstance(TestRegressionMetrics.assert_mape, numbers.Number)
        self.assertIsInstance(TestRegressionMetrics.assert_mape, float)
        assert TestRegressionMetrics.assert_mape >= 0
        
        
    def test_model_mse(self):
        
        TestRegressionMetrics.assert_mse = mm.model_mse(self.actual_mpg,self.predicted_mpg)
        self.assertEqual(len(self.actual_mpg), len(self.predicted_mpg))
        self.assertIsInstance(TestRegressionMetrics.assert_mse, numbers.Number)
        self.assertIsInstance(TestRegressionMetrics.assert_mse, float)
        assert TestRegressionMetrics.assert_mse >= 0
        
    def tearDown(self):
        print('Test Success!!!')
    
    @classmethod
    def tearDownClass(cls):
        print("Regressor Assessment: \n RMSE {} MAE: {} MAPE: {} MSE: {}".format(TestRegressionMetrics.assert_rmse,TestRegressionMetrics.assert_mae, 
              TestRegressionMetrics.assert_mape,TestRegressionMetrics.assert_mse))
unittest.main (argv =[''], verbosity=2, exit= False) 