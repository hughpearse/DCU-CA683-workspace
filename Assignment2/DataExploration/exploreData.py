import matplotlib.pyplot as plt
import csv
import pandas as pd

orders = pd.read_csv("input/orders.csv")
prior = orders['eval_set']=='prior'
ordersPrior = orders[prior]
counts = ordersPrior['order_number'].value_counts()
uniqueOrderNumbers = set(ordersPrior['order_number'])

plt.plot(list(uniqueOrderNumbers),counts)
plt.xlabel('Number of Orders')
plt.ylabel('Number of Customers', labelpad=-0.5)
plt.title('Number of Orders per Customer')
plt.savefig('foo.png')
