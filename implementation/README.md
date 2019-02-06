```bash
foo@bar:~$ virtualenv sandbox
foo@bar:~$ virtualenv -p $(which python3) sandbox
foo@bar:~$ source sandbox/bin/activate
foo@bar:~$ pip install pandas scipy numpy sklearn matplotlib
foo@bar:~$ chmod 700 ./modeller.py
foo@bar:~$ python3 ./modeller.py iris.csv
foo@bar:~$ deactivate
```

# Log output
```bash
(sandbox) foo@bar:~$ python3 ./modeller.py iris_with_missing_data.csv
Summary statistics class distribution:
virginica     50
setosa        50
versicolor    50
Name: Species, dtype: int64

setosa
Min Sepal.Length: 4.3
Max Sepal.Length: 5.8
Mean Sepal.Length: 5.0
Median Sepal.Length: 5.0
Min Sepal.Width: 2.3
Max Sepal.Width: 4.4
Mean Sepal.Width: 3.419148936170213
Median Sepal.Width: 3.4
Min Petal.Length: 1.0
Max Petal.Length: 1.9
Mean Petal.Length: 1.4693877551020411
Median Petal.Length: 1.5
Min Petal.Width: 0.1
Max Petal.Width: 0.6
Mean Petal.Width: 0.24599999999999997
Median Petal.Width: 0.2

versicolor
Min Sepal.Length: 4.9
Max Sepal.Length: 7.0
Mean Sepal.Length: 5.936
Median Sepal.Length: 5.9
Min Sepal.Width: 2.0
Max Sepal.Width: 3.4
Mean Sepal.Width: 2.7755102040816326
Median Sepal.Width: 2.8
Min Petal.Length: 3.0
Max Petal.Length: 5.1
Mean Petal.Length: 4.26
Median Petal.Length: 4.35
Min Petal.Width: 1.0
Max Petal.Width: 1.8
Mean Petal.Width: 1.3259999999999998
Median Petal.Width: 1.3

virginica
Min Sepal.Length: 4.9
Max Sepal.Length: 7.9
Mean Sepal.Length: 6.564583333333332
Median Sepal.Length: 6.45
Min Sepal.Width: 2.2
Max Sepal.Width: 3.8
Mean Sepal.Width: 2.9734693877551024
Median Sepal.Width: 3.0
Min Petal.Length: 4.5
Max Petal.Length: 6.9
Mean Petal.Length: 5.561224489795919
Median Petal.Length: 5.6
Min Petal.Width: 1.4
Max Petal.Width: 2.5
Mean Petal.Width: 2.0260000000000002
Median Petal.Width: 2.0

Means with NaN:
            Sepal Length  Sepal Width  Petal Length  Petal Width
setosa          5.000000     3.419149      1.469388     0.246000
versicolor      5.936000     2.775510      4.260000     1.326000
virginica       6.564583     2.973469      5.561224     2.026000
mean            5.833528     3.056043      3.763537     1.199333

Means without NaN:
            Sepal Length  Sepal Width  Petal Length  Petal Width
setosa          5.000000     3.413333      1.468889     0.248889
versicolor      5.942857     2.775510      4.267347     1.330612
virginica       6.580435     2.982609      5.541304     2.034783
mean            5.841097     3.057151      3.759180     1.204761

Chi-square of (means_with_nan,means_without_nan) p-value: 1.0
P-value was not significant, data is MCAR. Imputing missing values

Means after imputation:
            Sepal Length  Sepal Width  Petal Length  Petal Width
setosa          5.000000     3.419149      1.469388     0.246000
versicolor      5.936000     2.775510      4.260000     1.326000
virginica       6.564583     2.973469      5.561224     2.026000
mean            5.833528     3.056043      3.763537     1.199333

Calculating correlation between control variables:
              Sepal Length  Sepal Width  Petal Length  Petal Width
Sepal Length           NaN     0.122946      0.867964     0.819781
Sepal Width            NaN          NaN      0.427774     0.365409
Petal Length           NaN          NaN           NaN     0.962617
Petal Width            NaN          NaN           NaN          NaN

Dropping variable with high multicollinearity:  ['Petal Width']

Feature Importance (Chi^2): [ 8.59764474  2.43587713 92.04457581]
Features Selected: [ True False  True]
Selecting: ['Sepal Length' 'Petal Length']

Evaluating classifiers:
name,accuracy,std_dev
LR,0.916667,0.091287
LDA,0.966667,0.055277
KNN,0.941667,0.053359
CART,0.916667,0.052705
NB,0.900000,0.081650
SVM,0.958333,0.055902
MLP,0.950000,0.066667
RFC,0.891667,0.091667
GPC,0.941667,0.065085
ABC,0.641667,0.158333
QDA,0.958333,0.055902
SDG,0.766667,0.122474
GBC,0.916667,0.074536
NSVC,0.950000,0.040825

Selected classifier: LDA

LDA classifier validation test results:
Sepal Length,Petal Length,real-class,predicted-class
[5.8 1.2 'setosa' 'setosa']
[5.1 3.0 'versicolor' 'versicolor']
[6.6 4.4 'versicolor' 'versicolor']
[5.4 1.3 'setosa' 'setosa']
[7.9 6.4 'virginica' 'virginica']
[6.3 4.7 'versicolor' 'versicolor']
[6.9 5.1 'virginica' 'versicolor']  <== Misclassified
[5.1 1.9 'setosa' 'setosa']
[4.7 1.6 'setosa' 'setosa']
[6.9 5.7 'virginica' 'virginica']
[5.6 4.2 'versicolor' 'versicolor']
[5.4 1.7 'setosa' 'setosa']
[7.1 5.9 'virginica' 'virginica']
[6.4 4.5 'versicolor' 'versicolor']
[6.0 4.5 'versicolor' 'versicolor']
[4.4 1.3 'setosa' 'setosa']
[5.8 4.0 'versicolor' 'versicolor']
[5.6 4.5 'versicolor' 'versicolor']
[5.4 1.5 'setosa' 'setosa']
[5.0 1.2 'setosa' 'setosa']
[5.5 4.4 'versicolor' 'versicolor']
[5.4 4.5 'versicolor' 'virginica']  <== Misclassified
[6.7 5.0 'versicolor' 'versicolor']
[5.0 1.3 'setosa' 'setosa']
[7.2 6.0 'virginica' 'virginica']
[5.7 4.1 'versicolor' 'versicolor']
[5.5 1.4 'setosa' 'setosa']
[5.1 1.5 'setosa' 'setosa']
[6.1 4.7 'versicolor' 'versicolor']
[6.3 5.0 'virginica' 'virginica']
Accuracy: 0.9333333333333333
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        11
  versicolor       0.92      0.92      0.92        13
   virginica       0.83      0.83      0.83         6

   micro avg       0.93      0.93      0.93        30
   macro avg       0.92      0.92      0.92        30
weighted avg       0.93      0.93      0.93        30
```
