
from __future__ import division, print_function, absolute_import
 
import numpy as np
import random
import tensorflow as tf
from ae import Autoencoder
from sklearn.utils import shuffle

import pandas as pd

nm=495
nd=383
nc=5430

miRNAnumber=np.genfromtxt(r'miRNA number.txt',dtype=str,delimiter='\t')
diseasenumber=np.genfromtxt(r'disease number.txt',dtype=str,delimiter='\t')
DS1=np.loadtxt(r'disease semantic simialrity matrix 1.txt')
DS2=np.loadtxt(r'disease semantic simialrity matrix 2.txt')
DS=(DS1+DS2)/2
DS_weight=np.loadtxt(r'disease semantic simialrity weight matrix 1.txt')

FS=np.loadtxt(r'miRNA functional similarity matrix.txt')
FS_weight=np.loadtxt(r'miRNA functional similarity weight matrix.txt')


A = np.zeros((nm,nd),dtype=float)
ConnectData= np.loadtxt(r'known miRNA-disease assocaitions.txt',dtype=int)-1

for i in range(nc):
    A[ConnectData[i,0], ConnectData[i,1]] = 1
data0_index=np.argwhere(A==0)

def Getgauss_miRNA(adjacentmatrix,nm):
    KM=np.zeros((nm,nm),dtype=np.float32)
    gamma=1
    sum_norm=0                           
    for i in range(nm):
        sum_norm=np.linalg.norm(adjacentmatrix[i],ord=2)**2+sum_norm     
    gamma=gamma/(sum_norm/nm)
        
    for i in range(nm):
        for j in range(nm):
            if j<=i:
                KM[i,j]=np.exp(-gamma*(np.linalg.norm(adjacentmatrix[i]-adjacentmatrix[j]))**2)
    KM=KM+KM.T-np.eye(nm)
    return KM
    

def Getgauss_disease(adjacentmatrix,nd):
    KD=np.zeros((nd,nd),dtype=np.float32)
    gamma=1
    sum_norm=0
    for i in range(nd):
        sum_norm=np.linalg.norm(adjacentmatrix[:,i])**2+sum_norm    
    gamma=gamma/(sum_norm/nd)
        
    for i in range(nd):
        for j in range(nd):
            if j<=i:
                KD[i,j]=np.exp(-gamma*(np.linalg.norm(adjacentmatrix[:,i]-adjacentmatrix[:,j]))**2)
    KD=KD+KD.T-np.eye(nd)
    return KD

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]



def multilayer_perceptron(x, weights, biases):
    hidden_layer_one = tf.matmul(x, weights["W_layer_one"]) + biases["b_layer_one"]
        

    hidden_layer_one = tf.nn.relu(hidden_layer_one)
    hidden_layer_two = tf.matmul(hidden_layer_one, weights["W_layer_two"]) + biases["b_layer_two"]
    hidden_layer_two = tf.nn.relu(hidden_layer_two)
    hidden_layer_three = tf.matmul(hidden_layer_two, weights["W_layer_three"]) + biases["b_layer_three"]
    hidden_layer_three = tf.nn.relu(hidden_layer_three)
        

    out_layer = tf.matmul(hidden_layer_three, weights["W_out_layer"]) + biases["b_out_layer"]
    return out_layer

KM=Getgauss_miRNA(A,nm)                
KD=Getgauss_disease(A,nd)


SM=np.zeros((nm,nm),dtype=float)
for i in range(nm):
    for j in range(nm):
        if FS_weight[i,j]==1:                
            SM[i,j]=FS[i,j]
        else:
            SM[i,j]=KM[i,j]

SD=np.zeros((nd,nd),dtype=float)
for i in range(nd):
    for j in range(nd):
        if DS_weight[i,j]==1:
            SD[i,j]=DS[i,j]
        else:
            SD[i,j]=KD[i,j]



train_set=np.zeros((189585,878),dtype=float)  
for i in range(495):
    for j in range(383):
        T=[]
        T.extend(SM[i])     
        T.extend(SD[j])
        train_set[i*383+j]=T
        
training_epochs = 100
batch_size = 128
display_step = 1
stack_size = 3 
hidden_size = [512, 256, 128]
outputsize=2

sae = []
for i in range(stack_size):
    if i == 0:
        ae = Autoencoder(n_input = 878,                                                   
                        n_hidden = hidden_size[i],
                        transfer_function = tf.nn.tanh,
                        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001))
        ae._initialize_weights()
        sae.append(ae)
    else:
        ae = Autoencoder(n_input=hidden_size[i-1],
                         n_hidden=hidden_size[i],
                         transfer_function=tf.nn.tanh,
                         optimizer=tf.train.AdamOptimizer(learning_rate=0.001))
        ae._initialize_weights()
        sae.append(ae)
        
        
W = []
b = []
Hidden_feature = []         
X_train = np.array([0])

for j in range(stack_size):
    
    if j == 0:
        X_train = np.array(train_set)
 
    else:
        X_train_pre = X_train
        X_train = sae[j-1].transform(X_train_pre)
        
        Hidden_feature.append(X_train)
    
    for epoch in range(training_epochs):
        avg_cost = 0.                              
        total_batch = int(X_train.shape[1] / batch_size)
        
        for k in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
 
           
            cost = sae[j].partial_fit(batch_xs)
          
            avg_cost += cost / X_train.shape[1] * batch_size
 


    weight = sae[j].getWeights()

    W.append(weight)
    b.append(sae[j].getBiases())

Hidden_feature.append(sae[j].transform(X_train))
W1=W[0]
W2=W[1]
W3=W[2]
b1=b[0]
b2=b[1]
b3=b[2]                
            

Zheng=np.zeros((5430,878),dtype=float)  
for i in range(5430):
    zheng=[]
    zheng.extend(SM[ConnectData[i,0]])     
    zheng.extend(SD[ConnectData[i,1]])
    Zheng[i]=zheng
        
       
suiji=random.sample(list(data0_index),5430)
Fu=np.zeros((5430,878),dtype=float)
for i in range(5430):
    fu=[]
    fu.extend(SM[suiji[i][0]])
    fu.extend(SD[suiji[i][1]])
    Fu[i]=fu
    
       
labels=[]
for i in range(5430):
    labels.append([0,1])
for i in range(5430):
    labels.append([1,0])
labels=np.array(labels,dtype=int)
        
        
feature=np.vstack((Zheng,Fu))
feature=np.array(feature,dtype=float)
        
feature,labels=shuffle(feature,labels)


    
learning_rate = 0.01
iterations = 15
batch_number = 100
display_step = 1
 

X = tf.placeholder("float", [None, 878])  
Y = tf.placeholder("float", [None, 2])  
 

 
 
 

weights = {"W_layer_one":tf.Variable(W1),
           "W_layer_two":tf.Variable(W2),
           "W_layer_three":tf.Variable(W3),
           'W_out_layer':tf.Variable(tf.random_normal([hidden_size[2],2]))
    }
biases = {"b_layer_one":tf.Variable(b1),
        "b_layer_two":tf.Variable(b2),
        "b_layer_three":tf.Variable(b3),
        "b_out_layer":tf.Variable(tf.random_normal([2]))  
    } 
 

pred = multilayer_perceptron(X, weights, biases)

 

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred)) 


optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

y_score = tf.nn.softmax(logits=pred)


init = tf.global_variables_initializer()   
sess = tf.Session()
sess.run(init)

 
for iteration in range(iterations):
    
    
    train_cost=0
    batch_times=int(10860/batch_number)   
    
    for i in range(batch_times):
       
 
        start=(i*batch_number) % 10860 
        end=min(start+batch_number,10860)
       
        _,batch_cost=sess.run([optimizer,cost],feed_dict={X:feature[start:end],Y:labels[start:end]})
        
        
        train_cost+=batch_cost/batch_times
        
 
           
    
score_0=[]
henduo=np.zeros((data0_index.shape[0],878),dtype=float)
for i in range(data0_index.shape[0]):
    HENDUO=[]
    HENDUO.extend(SM[data0_index[i][0]])
    HENDUO.extend(SD[data0_index[i][1]])
    henduo[i]=HENDUO

zuizhong=y_score.eval(session=sess, feed_dict={X:henduo})
sco_henduo=np.array(zuizhong)
score_henduo_DNN=sco_henduo[:,1]         
score_0.extend(score_henduo_DNN)
score_0_sorted=sorted(score_0,reverse=True)


score_00=np.array(score_0)
score_0ranknumber =np.transpose(np.argsort(-score_00)) 


diseaserankname_pos =data0_index[score_0ranknumber,1]
diseaserankname = diseasenumber[diseaserankname_pos,1]
diseaserankname=diseaserankname.reshape(184155,)


miRNArankname_pos =data0_index[score_0ranknumber,0]
miRNArankname = miRNAnumber[miRNArankname_pos,1]
miRNArankname=miRNArankname.reshape(184155,)


score_0rank_pd=pd.Series(score_0_sorted)
diseaserankname_pd=pd.Series(diseaserankname)
miRNArankname_pd=pd.Series(miRNArankname)
prediction_0_out = pd.concat([diseaserankname_pd,miRNArankname_pd,score_0rank_pd],axis=1)
prediction_0_out.columns=['Disease','miRNA','Score']
prediction_0_out.to_excel(r'prediction result.xlsx', sheet_name='Sheet1',index=False)
               