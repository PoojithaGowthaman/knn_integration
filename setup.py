from setuptools import setup, find_packages

setup(
    name='ubc-knn',
    version=0.11,
    packages=find_packages(exclude=['test*']),
    license='MIT',
    description='KNN model in Python',
    long_description='Implementation of K nearest neighbors model in Python. Supports built-in tuning of k hyper-parameter using k-fold cross validation.',
    url='https://github.com/PoojithaGowthaman/knn_integration',
    author='Poojitha Gowthaman & Eric Phillips',
    author_email='ericphillips99@me.com'
)
