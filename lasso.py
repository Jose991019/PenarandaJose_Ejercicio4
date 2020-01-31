import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing 
import itertools 


%matplotlib inline

data = pd.read_csv('Cars93.csv')

Y = np.array(data['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
X = np.array(data[columns])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3)

                  
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

total = range(len(X_train))
x = range(1,12)
y = []
for i in x:
    r2 = []
    combinacion = itertools.combinations(total,i)
    for j in combinacion:
        lista = []
        for k in range(i):
            lista.append(j)
        X_analizar = X_train[lista]
        Y_analizar = Y_train[lista]
        regresion = sklearn.linear_model.LinearRegression()
        regresion.fit(X_analizar, Y_analizar)
        r2.append(regresion.score)
    y.append(r2)