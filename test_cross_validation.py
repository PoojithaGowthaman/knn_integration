import unittest
from KNN.modelling import generate_predictions as gp
from KNN.modelling import KNN_data_collection as knn
from KNN.assessment import cross_validation as cv
import numpy as np

class test_cross_validation(unittest.TestCase):

    def setUp(self):
        # Perform cross validation before each test
        self.knn_cv_regressor.perform_cv(self.k_values)
        self.knn_cv_classifier.perform_cv(self.k_values)

    @classmethod
    def setUpClass(self):
        # Set example k values
        self.k_values=[2,3,4,6,10,20]
        # Create KNN regressor, load CSV, perform train/test split
        self.knn_cv_regressor=cv.CvKNN('regressor')
        self.knn_cv_regressor.load_csv('datasets/auto_mpg.csv','mpg')
        self.knn_cv_regressor.train_test_split()
        # Create KNN classifier, load CSV, perform train/test split
        self.knn_cv_classifier=cv.CvKNN('classifier',8)
        self.knn_cv_classifier.load_csv('datasets/iris.csv','Species')
        self.knn_cv_classifier.train_test_split()

    def test_perform_cv(self):
        # Ensure that the number of results matches the number of k_values inputted
        self.assertEqual(len(self.knn_cv_regressor._CvKNN__k_results),len(self.k_values))
        self.assertEqual(len(self.knn_cv_classifier._CvKNN__k_results),len(self.k_values))

        # Ensure that all CV results are positive
        self.assertTrue(np.mean([i>0 for i in self.knn_cv_regressor._CvKNN__k_results]))
        self.assertTrue(np.mean([i>0 for i in self.knn_cv_classifier._CvKNN__k_results]))

        # Ensure that misclassification rate for classifier is <=1
        self.assertTrue(np.max(self.knn_cv_classifier._CvKNN__k_results)<=1)

    def test_get_cv_results(self):
        # Regressor
        # Check points on plot returned by get_cv_results function (ensure they match with k_values and k_results attributes and are on correct axis)
        cv_results_reg=self.knn_cv_regressor.get_cv_results()
        x_data_reg=cv_results_reg.lines[0].get_xdata()
        y_data_reg=cv_results_reg.lines[0].get_ydata()
        # Check that data from plot matches k_values and k_results attributes element-wise (if all match, mean of boolean array is 1)
        self.assertTrue(np.mean(x_data_reg==self.knn_cv_regressor._CvKNN__k_values))
        self.assertTrue(np.mean(y_data_reg==self.knn_cv_regressor._CvKNN__k_results))
        # Check that correct number of folds is in plot title
        plot_title_reg=cv_results_reg.get_title()
        num_folds_str_reg=str(self.knn_cv_regressor.num_folds)
        self.assertIn(num_folds_str_reg,plot_title_reg.split(' '))
        # Check that y label contains correct metric (MSE) for regressor
        plot_ylab_reg=cv_results_reg.get_ylabel()
        self.assertIn('MSE',plot_ylab_reg.split(' '))


        # Classifier
        # Check points on plot returned by get_cv_results function (ensure they match with k_values and k_results attributes and are on correct axis)
        cv_results_clf=self.knn_cv_classifier.get_cv_results()
        x_data_clf=cv_results_clf.lines[0].get_xdata()
        y_data_clf=cv_results_clf.lines[0].get_ydata()
        # Check that data from plot matches k_values and k_results attributes element-wise (if all match, mean of boolean array is 1)
        self.assertTrue(np.mean(x_data_clf==self.knn_cv_classifier._CvKNN__k_values))
        self.assertTrue(np.mean(y_data_clf==self.knn_cv_classifier._CvKNN__k_results))
        # Check that correct number of folds is in plot title
        plot_title_clf=cv_results_clf.get_title()
        num_folds_str_clf=str(self.knn_cv_classifier.num_folds)
        self.assertIn(num_folds_str_clf,plot_title_clf.split(' '))
        # Check that y label conrtains correct metric (Misclassification rate) for classifier
        plot_ylab_clf=cv_results_clf.get_ylabel()
        self.assertIn('Misclassification',plot_ylab_clf.split(' '))

    def test_get_best_k(self):
        # Check that value returned by get_best_k has the lowest associated value in result
        self.knn_cv_regressor.get_best_k()
        min_result_reg=np.min(self.knn_cv_regressor._CvKNN__k_results)
        min_position_reg=self.knn_cv_regressor._CvKNN__k_results.index(min_result_reg)
        true_best_k_reg=self.k_values[min_position_reg]
        self.assertEqual(self.knn_cv_regressor.k,true_best_k_reg)
        self.assertEqual(self.knn_cv_regressor.best_k,true_best_k_reg)

        self.knn_cv_classifier.get_best_k()
        min_result_clf=np.min(self.knn_cv_classifier._CvKNN__k_results)
        min_position_clf=self.knn_cv_classifier._CvKNN__k_results.index(min_result_clf)
        true_best_k_clf=self.k_values[min_position_clf]
        self.assertEqual(self.knn_cv_classifier.k,true_best_k_clf)
        self.assertEqual(self.knn_cv_classifier.best_k,true_best_k_clf)
#def
