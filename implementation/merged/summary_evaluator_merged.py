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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC, NuSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
import operator
import matplotlib  # noqa
matplotlib.use('TkAgg')  # noqa
import matplotlib.pyplot as plt  # noqa
import itertools

def main():
    df_1 = pandas.read_csv(sys.argv[1])
    df_2 = pandas.read_csv(sys.argv[2])
    summary(df_1,df_2)
    evaluator(df_1)
    exit(1)

def describeColumn(df):
    print("Min " + df.head(0).name + ": "  + str(df.min()))
    print("Max " + df.head(0).name + ": "  + str(df.max()))
    print("Mean " + df.head(0).name + ": "  + str(df.mean()))
    print("Median " + df.head(0).name + ": "  + str(df.median()))

def summary(df_1,df_2):
    df_collection = {}
    frames = [df_1, df_2]
    df_3 = pandas.concat(frames)

    df_collection[sys.argv[0]] = df_1
    df_collection[sys.argv[1]] = df_2
    df_collection['merged'] = df_3
    
    for key,val in df_collection.items():
        df = val
        print("\n================================")
        print("Summary statistics for: " + key)
        print("Class distribution:")
        print(df['Species'].value_counts())

        for species in df['Species'].unique():
            print("\n" + species)
            sp_df = df[df['Species']==species]
            column_names = list(sp_df.columns.values)
            column_names = column_names[:len(column_names)-1]
            for column_name in column_names:
                #print("-----" + column_name)
                if column_name != 'Species':
                    describeColumn(sp_df[column_name])

if len(sys.argv) != 3:
    print("Usage: python3 " + sys.argv[0] + " trainingData.csv testData.csv")


def evaluator(df_1):
    dataset = pandas.read_csv(sys.argv[1])
    dataset.fillna(dataset.mean(), inplace=True)
    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]

    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
        X, Y, test_size=validation_size, random_state=seed)

    chi2_selector = SelectKBest(score_func=chi2, k=2)
    chi2_selector.fit(X_train, Y_train)
    chi2_selector.fit_transform(X_train, Y_train)
    print("\nFeature Importance (Chi^2): " + str(chi2_selector.scores_))
    print("Features Selected: " + str(chi2_selector.get_support()))
    feature_names = dataset.iloc[:, chi2_selector.get_support()].columns.values
    print("Selecting: " + str(feature_names))
    X = chi2_selector.transform(X)
    X_train = chi2_selector.transform(X_train)
    X_validation = chi2_selector.transform(X_validation)

    scoring = 'accuracy'
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    models.append(('MLP', MLPClassifier(solver='lbfgs')))
    models.append(('RFC', RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1)))
    models.append(('GPC', GaussianProcessClassifier(1.0 * RBF(1.0))))
    models.append(('ABC', AdaBoostClassifier()))
    models.append(('QDA', QuadraticDiscriminantAnalysis()))
    models.append(('SDG', SGDClassifier(max_iter=1000, tol=0.05)))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('NSVC', NuSVC(probability=True, gamma='auto')))
    # evaluate each model in turn
    results = []
    names = []
    print("\nEvaluating classifiers:")
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
    print(feature_names[0]+","+feature_names[1]+",real-class,predicted-class")
    print(joined_testdata_w_predictions)
    print("Accuracy: " + str(accuracy_score(Y_validation, predictions)))
    print(classification_report(Y_validation, predictions))

    colors = {'virginica': 'red',
              'setosa': 'blue', 'versicolor': 'green'}
    plt.scatter(dataset[feature_names[0]], dataset[feature_names[1]],
                c=dataset["Species"].apply(lambda x: colors[x]))
    for row in joined_testdata_w_predictions:
        if row[2] != row[3]:
            plt.scatter(row[0], row[1], c='black', marker='x')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Iris dataset")
    plt.show()
    

if len(sys.argv) != 3:
    print("Usage: python3 " + sys.argv[0] + " trainingData.csv")

main()
