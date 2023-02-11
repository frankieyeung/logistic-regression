import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns = iris.feature_names)
y = pd.DataFrame(iris.target, columns = ['target'])
iris = pd.concat([X, y], axis = 1)
print(iris)

iris = iris[['sepal length (cm)', 'petal length (cm)', 'target']]
iris = iris[iris['target'].isin([0, 1])]
print(iris)

X = iris[['sepal length (cm)', 'petal length (cm)']]
y = iris[['target']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)

model = LogisticRegression()
model.fit(X_train_std, y_train['target'])

y_pred = model.predict(X_test_std)
print(y_test['target'].values)
print(y_pred)
print(model.score(X_test_std, y_test))