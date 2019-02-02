#!/bin/python3
import sys
import pandas
from pandas.plotting import scatter_matrix
import scipy
import numpy
import sklearn
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import operator


def main():
    dataset = pandas.read_csv(sys.argv[1])
    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
        X, Y, test_size=validation_size, random_state=seed)
    scoring = 'accuracy'
    models = []
    models.append(('LR', LogisticRegression(
        solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    # evaluate each model in turn
    results = []
    names = []
    print("Evaluating classifiers:")
    print("name,accuracy,std_dev")
    classifier_performance_dict = {}
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s,%f,%f" % (name, cv_results.mean(), cv_results.std())
        classifier_performance_dict[name] = cv_results.mean()
        print(msg)

    maxKey = max(classifier_performance_dict.items(),
                 key=operator.itemgetter(1))[0]
    classifier = ""
    modelDict = dict(models)
    classifier = modelDict[maxKey]

    print("\nSelected classifier: " + maxKey)
    classifier.fit(X_train, Y_train)
    predictions = classifier.predict(X_validation)
    joined_testdata = numpy.concatenate(
        (X_validation, numpy.reshape(Y_validation, (-1, 1))), axis=1)
    joined_testdata_w_predictions = numpy.concatenate(
        (joined_testdata, numpy.reshape(predictions, (-1, 1))), axis=1)
    print("\n" + maxKey + " classifier validation test results:")
    print("sl,sw,pl,pw,real-class,predicted-class")
    print(joined_testdata_w_predictions)
    print("Accuracy: " + str(accuracy_score(Y_validation, predictions)))
    print(classification_report(Y_validation, predictions))


if len(sys.argv) != 2:
    print("Usage: python3 " + sys.argv[0] + " trainingData.csv")
    exit(1)

main()
