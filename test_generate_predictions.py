import unittest
from KNN.modelling import generate_predictions as gp
from KNN.modelling import KNN_data_collection as knn
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

class test_generate_predictions(unittest.TestCase):

    def setUp(self):
        # "Train" models before each test
        # KNN regressor
        self.knn_regressor.load_csv('datasets/auto_mpg.csv','mpg')
        self.knn_regressor.train_test_split()
        # KNN classifier
        self.knn_classifier.load_csv('datasets/iris.csv','Species')
        self.knn_classifier.train_test_split()
        # Sklearn models (to compare against)
        self.sklearn_regressor.fit(self.knn_regressor.x,self.knn_regressor.y)
        self.sklearn_classifier.fit(self.knn_classifier.x,self.knn_classifier.y)

    @classmethod
    def setUpClass(self):
        # Create example observations for auto_mpg dataset
        self.reg_obs1=[4,97,46,1835,20.5]
        self.reg_obs2=[8,318,210,4382,13.5]
        self.reg_obs3=[-4,-97,-46,-1835,-20.5]
        # Create example observations for iris dataset
        self.clf_obs1=[5.2,3.4,1.4,0.2]
        self.clf_obs2=[7.2,3,5.8,1.6]
        self.clf_obs3=[-5.2,-3.4,-1.4,-0.2]

        # Create KNN regressor, load CSV, perform train/test split
        self.knn_regressor=knn.KNN('regressor')
        # Create KNN classifier, load CSV, perform train/test split
        self.knn_classifier=knn.KNN('classifier',4)

        # Compare our model's predictions with sklearn's KNN implementation
        # Set up and fit sklearn KNN regressor
        self.sklearn_regressor=KNeighborsRegressor(n_neighbors=3)
        # Set up and fit sklearn KNN classifier
        self.sklearn_classifier=KNeighborsClassifier(n_neighbors=4)

    def test_euclidean_distance(self):
        # Check if euclidean distance formula is implemented correctly
        distance_1=gp.euclidean_distance(self.reg_obs1,self.reg_obs2)
        true_distance_1=np.sqrt(((4-8)**2)+((97-318)**2)+((46-210)**2)+((1835-4382)**2)+((20.5-13.5)**2))
        self.assertEqual(distance_1,true_distance_1)

        distance_2=gp.euclidean_distance(self.reg_obs2,self.reg_obs3)
        true_distance_2=np.sqrt(((-4-8)**2)+((-97-318)**2)+((-46-210)**2)+((-1835-4382)**2)+((-20.5-13.5)**2))
        self.assertEqual(distance_2,true_distance_2)

        distance_3=gp.euclidean_distance(self.clf_obs1,self.clf_obs2)
        true_distance_3=np.sqrt(((5.2-7.2)**2)+((3.4-3)**2)+((1.4-5.8)**2)+((0.2-1.6)**2))
        self.assertEqual(distance_3,true_distance_3)

        pos_distance=gp.euclidean_distance(self.clf_obs1,self.clf_obs2)
        neg_distance=gp.euclidean_distance(self.clf_obs3,self.clf_obs2)
        self.assertNotEqual(neg_distance,pos_distance)

    def test_generate_prediction(self):
        # Test KNN regressor prediction against sklearn regressor prediction
        sklearn_reg_obs1_pred=self.sklearn_regressor.predict(np.array(self.reg_obs1).reshape(1,-1))
        knn_reg_obs1_pred=gp.generate_prediction(self.knn_regressor,self.reg_obs1,'all')
        self.assertEqual(knn_reg_obs1_pred,sklearn_reg_obs1_pred)

        sklearn_reg_obs2_pred=self.sklearn_regressor.predict(np.array(self.reg_obs2).reshape(1,-1))
        knn_reg_obs2_pred=gp.generate_prediction(self.knn_regressor,self.reg_obs2,'all')
        self.assertEqual(knn_reg_obs2_pred,sklearn_reg_obs2_pred)

        # Test KNN classifier prediction against sklearn classifier prediction
        sklearn_clf_obs1_pred=self.sklearn_classifier.predict(np.array(self.clf_obs1).reshape(1,-1))
        knn_clf_obs1_pred=gp.generate_prediction(self.knn_classifier,self.clf_obs1,'all')
        self.assertEqual(knn_clf_obs1_pred,sklearn_clf_obs1_pred)

        sklearn_clf_obs2_pred=self.sklearn_classifier.predict(np.array(self.clf_obs2).reshape(1,-1))
        knn_clf_obs2_pred=gp.generate_prediction(self.knn_classifier,self.clf_obs2,'all')
        self.assertEqual(knn_clf_obs2_pred,sklearn_clf_obs2_pred)

    def test_generate_predictions(self):
        # Test KNN regressor multiple predictions against sklearn regressor predictions
        reg_obs=np.array([self.reg_obs1,self.reg_obs2,self.reg_obs3])
        knn_reg_preds=gp.generate_predictions(self.knn_regressor,reg_obs,'all')
        sklearn_reg_preds=self.sklearn_regressor.predict(reg_obs)
        self.assertTrue(np.mean(knn_reg_preds==sklearn_reg_preds))
        # Check that length of inputted observations and results match
        self.assertEqual(len(reg_obs),len(knn_reg_preds))

        # Test KNN classifier multiple predictions against sklearn classifier predictions
        clf_obs=np.array([self.clf_obs1,self.clf_obs2,self.clf_obs3])
        knn_clf_preds=gp.generate_predictions(self.knn_classifier,clf_obs,'all')
        sklearn_clf_preds=self.sklearn_classifier.predict(clf_obs)
        self.assertTrue(np.mean(knn_clf_preds==sklearn_clf_preds))
        # Check that length of inputted observations and results match
        self.assertEqual(len(clf_obs),len(knn_clf_preds))

    def tearDown(self):
        print('Tests successfully completed.')
    @classmethod
    def tearDownClass(self):
        print('Class torn down.')
