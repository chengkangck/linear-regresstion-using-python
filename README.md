# linear-regresstion-using-python
## 1. The principle of linear regression
<!--
![image](https://github.com/chengkangck/linear-regresstion-using-python/blob/main/images/Mathematical%20representation%20of%20linear%20regression.png)
![image](https://github.com/chengkangck/linear-regresstion-using-python/blob/main/images/Least%20Squares%20Model.png)
![image](https://github.com/chengkangck/linear-regresstion-using-python/blob/main/images/problem.png)
![image](https://github.com/chengkangck/linear-regresstion-using-python/blob/main/images/Gradient%20descent.png)
-->
## 2. python implements linear regression
### 1. Basic matrix operations
```
# Author:kang cheng
import numpy as np
from numpy.linalg import inv
from numpy import dot
from numpy import mat

print('-------------给定矩阵A,B----------')
A = np.mat([1,1])
print ('A:\n',A)

B = mat([[1,2],[2,3]])
print ('B:\n',B)

print('--------------矩阵乘法-----------')
print('A.B:\n',dot(A,B))

print('--------------矩阵变形----------')
print('A.T:\n',A.T)
print('A.reshape(2,1):\n',A.reshape(2,1))
print('B.reshape(1,4):\n',B.reshape(1,4))
print('B的逆:\n',inv(B))
print('B[0,:]:\n',B[0,:])
print('B[:,0]:\n',B[:,0])
#print('A.B:',dot(B,A))
```
## 2. Realize the least squares method

```
# Author:kang cheng
import numpy as np
from numpy.linalg import inv
from numpy import dot
from numpy import mat

#y=2x
X = mat([1,2,3]).reshape(3,1)
Y = 2*X

#theta = (X'X)~-1X`Y
theta = dot(dot(inv(dot(X.T,X)),X.T),Y)
print(theta)
```
## 3. Realize the gradient descent method
```
import numpy as np
from numpy.linalg import inv
from numpy import dot
from numpy import mat

#y=2x
X = mat([1,2,3]).reshape(3,1)
Y = 2*X

#theta = theta - alpha*(theta*X -Y)*X
theta = 1.
alpha = 0.1
for i in range(100):
    theta = theta + np.sum(alpha * (Y- dot(X, theta))*X.reshape(1,3))/3.
print(theta)
```
## 4. Practical regression analysis
Calculate thera value by least square method

```
import numpy as np
from numpy.linalg import inv
from numpy import dot
from numpy import mat
import pandas as pd

dataset = pd.read_csv('data.csv')
# print(dataset)

temp = dataset.iloc[:, 2:5]
temp['X0'] = 1

X = temp.iloc[:, [3, 0, 1, 2]]
# print(X)

# Y = dataset.iloc[:,1]
# print(Y)

Y = dataset.iloc[:,1].values.reshape(200,1)#Y需要转置
# # 通过最小二乘法(向量法)算theta
theta = dot(dot(inv(dot(X.T, X)),X.T), Y)
print(theta)
```

Calculate thera value by gradient descent method

```
import numpy as np
from numpy.linalg import inv
from numpy import dot
from numpy import mat
import pandas as pd

dataset = pd.read_csv('data.csv')
# print(dataset)

temp = dataset.iloc[:, 2:5]
temp['X0'] = 1

X = temp.iloc[:, [3, 0, 1, 2]]
# print(X)

# Y = dataset.iloc[:,1]
# print(Y)

Y = dataset.iloc[:,1].values.reshape(200,1)#Y需要转置
# # 通过最小二乘法(向量法)算theta
theta = dot(dot(inv(dot(X.T, X)),X.T), Y)
print(theta)

# 通过梯度下降方法算theta
theta = np.array([1., 1., 1., 1.]).reshape(4, 1)
alpha = 0.1
temp = theta #使用缓存，使得梯度下降的时候更新
#200一般是lenth(Y)得到
# X0 = X.iloc[:, 0].reshape(200, 1)
# X1 = X.iloc[:, 1].reshape(200, 1)
# X2 = X.iloc[:, 2].reshape(200, 1)
# X3 = X.iloc[:, 3].reshape(200, 1)

# reshape 运行报错的话，是因为在pandas里面已经过时
X0 = X.iloc[:, 0].values.reshape(200, 1)
X1 = X.iloc[:, 1].values.reshape(200, 1)
X2 = X.iloc[:, 2].values.reshape(200, 1)
X3 = X.iloc[:, 3].values.reshape(200, 1)

# 同步更新
for i in range(1000):
    temp[0] = theta[0] + alpha*np.sum((Y- dot(X, theta))*X0)/200.
    temp[1] = theta[1] + alpha*np.sum((Y- dot(X, theta))*X1)/200.
    temp[2] = theta[2] + alpha*np.sum((Y- dot(X, theta))*X2)/200.
    temp[3] = theta[3] + alpha*np.sum((Y- dot(X, theta))*X3)/200.
    theta = temp
print(theta)
```
