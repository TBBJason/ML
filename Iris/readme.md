# Notes

## requirements/dependencies
- It's a good habit to start using requirements.txt, inputting all the dependencies there and then installing through
- pip install -r requirements.txt


## Univariate/Multivariate Plots
- Univariate plots better understand each attribute
- Multivariate plots better understand the relationship between attributes

## Test Harness
- The general form for splitting our training data and validation data is as follows:
- X = array[:, 0:n] where n represents the variable we are trying to classify
- Y = array[:, n] as an array of solutions that will map out to our X
### setting up the stratified cross validation:
Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=a, randomstate=b)