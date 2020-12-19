from setuptools import setup, find_packages

setup(
    name='ubc-knn',
    version=0.1,
    packages=find_packages(exclude=['test*']),
    license='MIT',
    description='Implementation of KNN model in Python. Includes built-in support for cross validation.',
    url='https://github.com/PoojithaGowthaman/knn_integration',
    author='Poojitha Gowthaman & Eric Phillips',
    author_email='ericphillips99@me.com'
)
