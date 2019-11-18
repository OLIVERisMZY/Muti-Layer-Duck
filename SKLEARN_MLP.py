import math
import pandas as pd
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv('D:\可爱臭鸭鸭\duck_data.csv')#需要添加目标文件目录
train_data = np.array(df.iloc[:,1:10])
attributeMap={}
attributeMap['white']=1
attributeMap['yellow']=0.5
attributeMap['black']=0
#====================
attributeMap['curl']=1
attributeMap['common']=0.5
attributeMap['straight']=0
#====================
attributeMap['dull']=0
attributeMap['loud']=0.5
attributeMap['clear']=1
#====================
attributeMap['hard']=0
attributeMap['common']=0.5
attributeMap['soft']=1
#====================
attributeMap['small']=0
attributeMap['common']=0.5
attributeMap['big']=1
attributeMap['ugly']=0
attributeMap['cute']=1
attributeMap['no']=0
attributeMap['yes']=1
#============数据化=================
for i in range(len(train_data)):
  for j in range(len(train_data[0])):
      if j != 6 and j != 7:
        train_data[i,j]=attributeMap[train_data[i,j]]
#============归一化=================
food=train_data[:,6]
health=train_data[:,7]
food_min=np.min(food)
food_max=np.max(food)
health_min=np.min(health)
health_max=np.max(health)
for i in range(len(food)):
    food[i]=( food[i]-food_min)/(food_max-food_min)
    food[i]=round( food[i],2)
for i in range(len(health)):
    health[i] = (health[i]- health_min) / (health_max - health_min)
    health[i] = round(health[i],2)

label=(train_data[:,8]).tolist()
train_data=np.delete(train_data,8,1)
train_data=train_data.tolist()
m,n=np.shape(train_data)
#===============数据划分========================================
x_train,x_test,y_train,y_test = train_test_split(train_data,label,test_size=0.2,random_state=0)
print(x_train)
print(y_train)
clf= MLPClassifier(hidden_layer_sizes=[9,9,9,9],max_iter=20000,learning_rate='constant', alpha=1e-5, learning_rate_init=0.0001,activation='relu',solver ='adam',verbose=True)
clf.fit(x_train, y_train)
print('模型得分:{:.2f}'.format(clf.score(x_test,y_test)))