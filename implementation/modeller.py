#!/bin/python3
import sys
import pandas
from pandas.plotting import scatter_matrix
import scipy
from scipy.stats import chisquare
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
from sklearn.ensemble import VotingClassifier
import operator
import matplotlib  # noqa
matplotlib.use('TkAgg')  # noqa
import matplotlib.pyplot as plt  # noqa
import itertools

colNames = ['Sepal Length', 'Sepal Width',
            'Petal Length', 'Petal Width', 'Species']
colNums = [0, 1, 2, 3, 4]


def describeColumn(df):
    print("Min " + df.head(0).name + ": " + str(df.min()))
    print("Max " + df.head(0).name + ": " + str(df.max()))
    print("Mean " + df.head(0).name + ": " + str(df.mean()))
    print("Median " + df.head(0).name + ": " + str(df.median()))


def generateSummary(df):
    print("Summary statistics class distribution: ")
    print(df[colNames[colNums[4]]].value_counts())

    for species in df[colNames[colNums[4]]].unique():
        print("\n" + species)
        sp_df = df[df[colNames[colNums[4]]] == species]
        column_names = list(sp_df.columns.values)
        column_names = column_names[:len(column_names)-1]
        for column_name in column_names:
            if column_name != colNames[colNums[4]]:
                describeColumn(sp_df[column_name])


def main():
    df = pandas.read_csv(sys.argv[1])
    generateSummary(df)
    df.columns = colNames
    # Means with NaNs
    means_with_nan = df.groupby(
        colNames[colNums[4]]).apply(lambda x: x.mean())
    means_with_nan = means_with_nan.append(pandas.DataFrame(
        means_with_nan.mean(numeric_only=True)).T)
    means_with_nan.rename(index={0: 'mean'}, inplace=True)
    print("\nMeans with NaN:")
    print(means_with_nan)
    means_with_nan = means_with_nan.values.flatten()
    # Means without NaNs
    means_without_nan = df.dropna().groupby(
        colNames[colNums[4]]).apply(lambda x: x.mean())
    means_without_nan = means_without_nan.append(pandas.DataFrame(
        means_without_nan.mean(numeric_only=True)).T)
    means_without_nan.rename(index={0: 'mean'}, inplace=True)
    print("\nMeans without NaN:")
    print(means_without_nan)
    means_without_nan = means_without_nan.values.flatten()
    # MCAR decision
    cs = chisquare(means_with_nan, means_without_nan)
    print("\nChi-square of (means_with_nan,means_without_nan) p-value:", cs.pvalue)
    if(cs.pvalue > 0.05):
        # CMAR - replace NaN by class mean
        print("P-value was not significant, data is MCAR. Imputing missing values")
        df[colNames[colNums[0]]] = df.groupby(colNames[colNums[4]])[
            colNames[colNums[0]]].apply(lambda x: x.fillna(x.mean()))
        df[colNames[colNums[1]]] = df.groupby(colNames[colNums[4]])[
            colNames[colNums[1]]].apply(lambda x: x.fillna(x.mean()))
        df[colNames[colNums[2]]] = df.groupby(colNames[colNums[4]])[
            colNames[colNums[2]]].apply(lambda x: x.fillna(x.mean()))
        df[colNames[colNums[3]]] = df.groupby(colNames[colNums[4]])[
            colNames[colNums[3]]].apply(lambda x: x.fillna(x.mean()))
        print("\nMeans after imputation:")
        means_after_impute = df.groupby(
            colNames[colNums[4]]).apply(lambda x: x.mean())
        means_after_impute = means_after_impute.append(
            pandas.DataFrame(means_after_impute.mean(numeric_only=True)).T)
        means_after_impute.rename(index={0: 'mean'}, inplace=True)
        print(means_after_impute)
    else:
        # NMAR - drop rows with NaN
        print("P-value was significant, data is NMAR. Dropping rows with missing values")
        df.dropna(inplace=True)
    # Drop highly correlated control variables
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(numpy.triu(
        numpy.ones(corr_matrix.shape), k=1).astype(numpy.bool))
    print("\nCalculating correlation between control variables:")
    print(upper_triangle)
    to_drop = [column for column in upper_triangle.columns if any(
        upper_triangle[column] > 0.95)]
    print("\nDropping variable with high multicollinearity: ", to_drop)
    df = df.drop(to_drop, axis=1)
    for i, x in enumerate(colNames):
        if x in to_drop:
            colNames.remove(x)
            colNums.remove(len(colNums)-1)
    # Create A/B split of data
    array = df.values
    X = array[:, 0:3]
    Y = array[:, 3]
    validation_size = 0.20
    seed = 1
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
        X, Y, test_size=validation_size, random_state=seed)
    # Select variables to predict response
    chi2_selector = SelectKBest(score_func=chi2, k=2)
    chi2_selector.fit(X_train, Y_train)
    chi2_selector.fit_transform(X_train, Y_train)
    print("\nFeature Importance (Chi^2): " + str(chi2_selector.scores_))
    feature_names = df.iloc[:, chi2_selector.get_support()].columns.values
    print("Selecting features: " + str(feature_names))
    X = chi2_selector.transform(X)
    X_train = chi2_selector.transform(X_train)
    X_validation = chi2_selector.transform(X_validation)
    # Instiantiate model list
    scoring = 'accuracy'
    models = []
    models.append(('LR', LogisticRegression(
        solver='liblinear', multi_class='ovr')))
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
    # Combine classifiers using a vote
    classifier = VotingClassifier(models)
    classifier.fit(X_train, Y_train)
    results = model_selection.cross_val_score(
        classifier, X_train, Y_train, cv=kfold)
    # Execute some tests
    predictions = classifier.predict(X_validation)
    joined_testdata = numpy.concatenate(
        (X_validation, numpy.reshape(Y_validation, (-1, 1))), axis=1)
    joined_testdata_w_predictions = numpy.concatenate(
        (joined_testdata, numpy.reshape(predictions, (-1, 1))), axis=1)
    print("\nEnselble (vote) classifier validation test results:")
    print(feature_names[0]+","+feature_names[1]+",real-class,predicted-class")
    for row in joined_testdata_w_predictions:
        if row[2] != row[3]:
            print(row, " <== Misclassified")
        else:
            print(row)
    print("Accuracy: " + str(accuracy_score(Y_validation, predictions)))
    print(classification_report(Y_validation, predictions))
    # Visualise results
    colors = {'virginica': 'red',
              'setosa': 'blue', 'versicolor': 'green'}
    plt.scatter(df[feature_names[0]], df[feature_names[1]],
                c=df[colNames[colNums[3]]].apply(lambda x: colors[x]))
    for row in joined_testdata_w_predictions:
        if row[2] != row[3]:
            plt.scatter(row[0], row[1], c='black', marker='x')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Iris dataset")
    plt.show()


if len(sys.argv) != 2:
    print("Usage: python3 " + sys.argv[0] + " iris.csv")
    sys.exit(1)

main()
