```bash
foo@bar:~$ virtualenv sandbox
foo@bar:~$ virtualenv -p $(which python3) sandbox
foo@bar:~$ source sandbox/bin/activate
foo@bar:~$ pip install pandas scipy numpy sklearn matplotlib
foo@bar:~$ chmod 700 ./evaluator.py
foo@bar:~$ python3 ./evaluator.py trainingData.csv
foo@bar:~$ python3 ./evaluator.py trainingData.csv
foo@bar:~$ deactivate
```
