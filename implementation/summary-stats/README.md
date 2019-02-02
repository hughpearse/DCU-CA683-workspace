```bash
foo@bar:~$ virtualenv sandbox
foo@bar:~$ virtualenv -p $(which python3) sandbox
foo@bar:~$ source sandbox/bin/activate
foo@bar:~$ pip install pandas
foo@bar:~$ chmod 700 ./summary.py
foo@bar:~$ python3 ./summary.py trainingData.csv testData.csv
foo@bar:~$ deactivate
```
