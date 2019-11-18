import math
import pandas as pd
import numpy as np
import random
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
#============提取去除标签=================
row_rand_array = np.arange(train_data.shape[0])
np.random.shuffle(row_rand_array)
row_rand = train_data[row_rand_array[0:10]]#抽取12条数据训练
rest_rand = train_data[row_rand_array[10:17]]#抽取12条数据训练
train_data=row_rand
test_data=rest_rand
test_label=test_data[:,8]
train_label=train_data[:,8]
train_data=np.delete(train_data,8,1)#第三个数1表示列，0表示行
test_data=np.delete(test_data,8,1)#第三个数1表示列，0表示行
m,n=np.shape(train_data)
c,d=np.shape(test_data)
#===============网络初始化===================
d=n   #输入向量维度
l=1   #输出向量维度
q=d+1   #隐藏层神经元个数
# 输入层到隐藏层的神经元的阈值
theta=np.random.random(size=(1,l))
# 隐藏层到输出层的神经元的阈值
gamma=np.random.random(size=(1,q))
# v = d*q 输入层与隐藏层之间的权重
v=np.random.random(size=(d,q))
# w = q*l 隐藏层与输出层之间的权重
w=np.random.random(size=(q,l))
#===============激活函数定义===================
def sigmoid(iX):#iX is a matrix with a dimension
    for i in range(len(iX[0])):
        iX[0,i] = 1 / (1 + math.exp(-iX[0,i]))
    return iX
eta = 0.001  # 学习率
maxIter = 10000  # 最大训练次数
while(maxIter>0):
    maxIter-=1
    sumE=0
    for i in range(m):
        alpha=np.dot(train_data[i].reshape(1,d),v)#shape=(1*d)*(d*q)=1*q
        b=sigmoid(alpha-gamma)#b=f(alpha-gamma), shape=1*q
        beta=np.dot(b,w)#shape=(1*q)*(q*l)=1*l
        predictY=sigmoid(beta-theta)   #shape=1*l ,p102--5.3
        E = sum((predictY-train_label[i])*(predictY-train_label[i]))/2    #5.4
        sumE+=E#5.16
        sumE=sumE/(i+1)
        #p104
        g=predictY*(1-predictY)*(train_label[i]-predictY)#shape=1*l p103--5.10
        e=b*(1-b)*((np.dot(w,g.T)).T) #shape=1*q , p104--5.15
        mid=eta*np.dot(b.T,g)#5.11
        w=w+mid
        mid=eta*g#5.12
        theta=theta-mid
        mid=eta*np.dot((train_data[i].reshape(d,1)),e)#5.13
        v=v+mid
        mid=eta*e#5.14
        gamma=gamma-mid
    print('loss='+str(np.round(sumE[0],4)))

def predict(data):
    count=len(data)
    predictY=[]
    for i in range(len(data)):
      alpha = np.dot(data[i].reshape(1,d), v)  # p101 line 2 from bottom, shape=m*q
      b = sigmoid(alpha - gamma)  # b=f(alpha-gamma), shape=m*q
      beta = np.dot(b, w)  # shape=(m*q)*(q*l)=m*l
      preY = sigmoid(beta - theta)  # shape=m*l ,p102--5.3
      predictY.append(preY)
    return predictY

predictY=predict(test_data)
wucha=0
for i in range(len(predictY)):
    if predictY[i]>0.5:
        predictY[i]=1
    else:
        predictY[i]=0
    if predictY[i]!=test_label[i]:
        wucha+=1
p=1-(wucha/len(predictY))
print(predictY)
print(test_label)
print('模型正确率为'+str(p*100)+'%')
