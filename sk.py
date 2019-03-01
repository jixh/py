from kanren import run,var,fact
from kanren.assoccomm import eq_assoccomm as eq
from kanren.assoccomm import commutative,associative
add = 'add'
mul = 'mul'
fact(commutative,mul)
fact(commutative,add)
fact(associative,mul)
fact(associative,add)
a,b = var('a'),var('b')
original_pattern = (mul,(add,5,a),b)
exp1 = (mul,2,(add,5,1))
exp2 = (add,5,(mul,8,1))

print(run(0,(a,b),eq(original_pattern,exp1)))
print(run(0,(a,b),eq(original_pattern,exp2)))




#import numpy as np
#from sklearn import preprocessing
#input_data = np.array([[2.1, -1.9, 5.5],
#                      [-1.5, 2.4, 3.5],
#                      [0.5, -7.9, 5.6],
#                      [5.9, 2.3, -5.8]])
#data_binarized = preprocessing.Binarizer(threshold = 0).transform(input_data)
#print("\nBinarized data:\n", data_binarized)
#print("Mean = ", input_data.mean(axis = 0))
#print("Std deviation = ", input_data.std(axis = 0))


#from sklearn import preprocessing
#data = [[0, 0], [0, 0], [1, 1], [1, 1]]
## 1. 基于mean和std的标准化
#scaler = preprocessing.StandardScaler().fit(train_data)
#scaler.transform(train_data)
#scaler.transform(test_data)
#
## 2. 将每个特征值归一化到一个固定范围
#scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_data)
#scaler.transform(train_data)
#scaler.transform(test_data)
##feature_range: 定义归一化范围，注用（）括起来



#from sklearn.datasets.samples_generator import make_classification
#X, y = make_classification(n_samples=6, n_features=5, n_informative=2,
#                           n_redundant=2, n_classes=2, n_clusters_per_class=2, scale=1.0,
#                           random_state=20)
#for x_,y_ in zip(X,y):
#    print(y_,end=': ')
#    print(x_)


#from sklearn import datasets
#boston = datasets.load_boston()
#_data = boston.data()
#_target = boston.target()
#_feature_name = boston.feature_names()
#print(boston.data)

#import sklearn
#from sklearn.datasets import load_iris
#iris = load_iris()

#from sklearn import datasets
#iris = datasets.load_iris()
#print(iris)
#x = iris.data
#y = iris.target
#
#for x_,y_ in zip(x,y):
#  print(y_,end=': ')
#  print(x_)


