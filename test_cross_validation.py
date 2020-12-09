import unittest
from KNN.modelling import generate_predictions as gp
from KNN.modelling import KNN_data_collection as knn
from KNN.assessment import cross_validation as cv
import numpy as np

class test_cross_validation(unittest.TestCase):

    def setUp(self):
        self.k_values=[2,3,4,6,10,20]

    @classmethod
    def setUpClass(self):
        # Create KNN regressor, load CSV, perform train/test split
        self.knn_cv_regressor=cv.CvKNN('regressor')
        self.knn_cv_regressor.load_csv('datasets/auto_mpg.csv','mpg')
        self.knn_cv_regressor.train_test_split()
        # Create KNN classifier, load CSV, perform train/test split
        self.knn_cv_classifier=cv.CvKNN('classifier')
        self.knn_cv_classifier.load_csv('datasets/iris.csv','Species')
        self.knn_cv_classifier.train_test_split()

    def test_perform_cv(self):
        # Ensure that the number of results matches the number of k_values inputted
        self.knn_cv_regressor.perform_cv(self.k_values)
        self.assertEqual(len(self.knn_cv_regressor._CvKNN__k_results),len(self.k_values))

        self.knn_cv_classifier.perform_cv(self.k_values)
        self.assertEqual(len(self.knn_cv_classifier._CvKNN__k_results),len(self.k_values))

        # Ensure that all CV results are positive
        self.assertTrue(np.mean([i>0 for i in self.knn_cv_regressor._CvKNN__k_results]))
        self.assertTrue(np.mean([i>0 for i in self.knn_cv_classifier._CvKNN__k_results]))

        # Ensure that misclassification rate for classifier is <=1
        self.assertTrue(np.max(self.knn_cv_classifier._CvKNN__k_results)<=1)

    def test_results(self):
        # Check that value returned has the lowest associated value in result
        self.knn_cv_regressor.perform_cv(self.k_values)
        self.knn_cv_regressor.get_best_k()
        min_result=np.min(self.knn_cv_regressor._CvKNN__k_results)
        min_position=self.knn_cv_regressor._CvKNN__k_results.index(min_result)
        true_best_k=self.k_values[min_position]
        self.assertEqual(self.knn_cv_regressor.k,true_best_k)

        self.knn_cv_classifier.perform_cv(self.k_values)
        self.knn_cv_classifier.get_best_k()
        min_result=np.min(self.knn_cv_classifier._CvKNN__k_results)
        min_position=self.knn_cv_classifier._CvKNN__k_results.index(min_result)
        true_best_k=self.k_values[min_position]
        self.assertEqual(self.knn_cv_classifier.k,true_best_k)
