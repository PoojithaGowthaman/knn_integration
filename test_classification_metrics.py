import unittest
from KNN.modelling import KNN_data_collection as knn
from KNN.assessment import model_metrics as mm
from KNN.modelling import generate_predictions as gp
import numpy as np
import random
import numbers

class KnnValueError(Exception):
    pass
class InvalidModel(Exception):
    pass


class TestClassificationMetrics(unittest.TestCase):
    accuracy, to_predict, assert_accuracy,assert_misclassify,assert_num_correct,num_incorrect,assert_num_incorrect =0,0,0,0,0,0, 0
    test_knn_classifier,k = 0, 0#Class or Static Variable, will be accessed using Class name
    model_type = " "
    @classmethod
    def setUpClass(cls):
        try:
            TestClassificationMetrics.test_knn_classifier = knn.KNN("classifier",4)
            if TestClassificationMetrics.k < 0 and TestClassificationMetrics.k > 10:
                raise KnnValueError()
            if TestClassificationMetrics.model_type!= "regressor" or TestClassificationMetrics.model_type!="classifier":
                raise InvalidModel()
        except KnnValueError:
            print('K nearest neighbors is advisable to a maximum of 10 for statistical effiency')
        except InvalidModel:
            print('Invalid Model type, We are perfoming KNN Regression or KNN Classification')
        
        
        try:
            TestClassificationMetrics.test_knn_classifier.load_csv('datasets/iris.csv','Species')
        except FileNotFoundError:
            print('Dataset not found, check datasets folder.')
        
        try:
            TestClassificationMetrics.test_knn_classifier.train_test_split(0.2)
        except ValueError:
            print('Enter a value between 0.1 to 0.9 as test-train split percentage')
        
        print('from setUpClass')
    def setUp(self):
        try:
            self.to_predict=np.array([[6.3,2.3,4.4,1.3],[6.8,3.2,5.9,2.3],[4.3,3,1.1,0.1]])
        except IndexError:
            print('Ensure predictor array lengths(x) are the same as response(y)')
        
        gp.generate_predictions(TestClassificationMetrics.test_knn_classifier,self.to_predict,'all')
        
    def test_accuracy(self):
        self.accuracy = 0.6666666666666666
        TestClassificationMetrics.assert_accuracy = mm.model_accuracy(['Iris-virginica', 'Iris-virginica', 'Iris-setosa'],gp.generate_predictions(TestClassificationMetrics.test_knn_classifier,self.to_predict,'all'))

        self.assertEqual(self.accuracy, self.assert_accuracy)
        self.assertIsInstance(TestClassificationMetrics.assert_accuracy, numbers.Number)
        self.assertIsInstance(TestClassificationMetrics.assert_accuracy, float)
        assert TestClassificationMetrics.assert_accuracy >= 0
        
    def test_model_misclassification(self):
        self.misclassify = 0.3333333333333333
        try:
            TestClassificationMetrics.assert_misclassify = mm.model_misclassification(['Iris-virginica', 'Iris-virginica', 'Iris-setosa'],gp.generate_predictions(TestClassificationMetrics.test_knn_classifier,self.to_predict,'all'))
        except TypeError:
            print ('Predictiors should be of string data type')

        self.assertEqual(self.assert_misclassify, self.misclassify)
        self.assertIsInstance(TestClassificationMetrics.assert_misclassify, numbers.Number)
        self.assertIsInstance(TestClassificationMetrics.assert_misclassify, float)
        assert TestClassificationMetrics.assert_misclassify >= 0
        
    def test_model_num_correct(self):
        self.num_correct = 2
        try:
            TestClassificationMetrics.assert_num_correct = mm.model_num_correct(['Iris-virginica', 'Iris-virginica', 'Iris-setosa'],gp.generate_predictions(TestClassificationMetrics.test_knn_classifier,self.to_predict,'all'))
        except TypeError:
            print('Predictiors should be of string data type')
    
        self.assertEqual(self.assert_num_correct, self.num_correct)
        self.assertIsInstance(TestClassificationMetrics.assert_num_correct, numbers.Number)
        self.assertNotIsInstance(TestClassificationMetrics.assert_num_correct, float)
        assert TestClassificationMetrics.assert_num_correct >= 0
        
    def test_model_num_incorrect(self):
        self.num_incorrect = 2
        try: 
            TestClassificationMetrics.assert_num_incorrect = mm.model_num_incorrect(['Iris-virginica', 'Iris-virginica', 'Iris-setosa'],gp.generate_predictions(TestClassificationMetrics.test_knn_classifier,self.to_predict,'all'))
        except TypeError:
            print('Predictiors should be of string data type')
            
        self.assertNotEqual(self.assert_num_incorrect, self.num_incorrect)
        self.assertIsInstance(TestClassificationMetrics.assert_num_incorrect, numbers.Number)
        self.assertNotIsInstance(TestClassificationMetrics.assert_num_incorrect, float)
        assert TestClassificationMetrics.assert_num_incorrect >= 0
    
    def tearDown(self):
        print('Test Success!!!')
        
    @classmethod
    def tearDownClass(cls):
        print("Classifier Assessment: \n Accuracy: {} Misclassification Rate: {} Numbers classified Correctly: {} Numbers classified Incorrectly: {}"
              .format(TestClassificationMetrics.assert_accuracy,TestClassificationMetrics.assert_misclassify,
              TestClassificationMetrics.assert_num_correct, TestClassificationMetrics.assert_num_incorrect))

unittest.main (argv =[''], verbosity=2, exit= False)