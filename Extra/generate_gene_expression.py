from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
pd.options.display.width = 0

X, y = make_regression(n_samples=539, n_features=100, n_informative=10)
X += 20  # shift the distributions of genes. The standard deviation remains the same
X += np.random.normal(loc=0, scale=0.3, size=X.shape)  # add gaussian noise
y += 800
data_joined = pd.DataFrame(np.c_[X, y],
                           columns=['Gene '+ str(x) for x in range(1, 101)] + ['Phenotype'])
print(data_joined.describe())

x_std = StandardScaler().fit_transform(X)
LR_1 = LinearRegression()
scores = cross_val_score(LR_1, x_std, y, cv=5, scoring='r2')
print(np.mean(scores))

data_joined.to_csv('simulated_gene_expression.csv', index=False)


