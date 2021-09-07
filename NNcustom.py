# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:11:31 2021

@author: Labib Rahman & MD Zafore Sadek
"""

import numpy as np
import matplotlib.pyplot as plt



#### Importing the Data

path=r'C:\Users\USER\Downloads\Iris.csv'

numpy_array = np.genfromtxt(path,delimiter=",", skip_header=1,encoding="utf8",dtype=np.str, usecols=(1,2,3,4))

np.random.seed(42)
feature_set=numpy_array.reshape(150,4)
feature_set=feature_set.astype('float64')
rng_state = np.random.get_state()
np.random.shuffle(feature_set)
training_set=feature_set[:100]
test_set=feature_set[100:150]




####   Iris-setosa=0  Iris-versicolor=1 Iris-virginica=2

labels=np.array([0]*50 + [1]*50 + [2]*50)
np.random.set_state(rng_state)
np.random.shuffle(labels)
training_labels=labels[:100]
test_labels=labels[100:150]

one_hot_labels = np.zeros((100, 3))
one_hot_labels_t = np.zeros((50, 3))

for i in range(100):
    one_hot_labels[i, training_labels[i]] = 1
for i in range(50):
    one_hot_labels_t[i, test_labels[i]] = 1    

plt.figure(figsize=(10,7))
plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap='plasma', s=200, alpha=0.7)
plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

attributes = training_set.shape[1]
hidden_nodes = 8
output_labels = 3

wh = np.random.rand(attributes,hidden_nodes)
bh = np.random.randn(hidden_nodes)

wo = np.random.rand(hidden_nodes,output_labels)
bo = np.random.randn(output_labels)
lr = 0.01

error_cost = []
epoch=0
epochs=[]
Accuracy=[]


print("Training Phase")
for epoch in range(5000): 
    
############# Feedforward
    
    # Phase 1
    zh = np.dot(training_set, wh) + bh
    ah = sigmoid(zh)
    
    # Phase 2
    zo = np.dot(ah, wo) + bo
    ao = softmax(zo)

########## Back Propagation

########## Phase 1

    dcost_dzo = ao - one_hot_labels
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

    dcost_bo = dcost_dzo

########## Phases 2

    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = training_set
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    dcost_bh = dcost_dah * dah_dzh

    # Update Weights ================

    wh -= lr * dcost_wh
    bh -= lr * dcost_bh.sum(axis=0)

    wo -= lr * dcost_wo
    bo -= lr * dcost_bo.sum(axis=0)

    if epoch % 1000 == 0:
        loss=np.sum(-one_hot_labels * np.log(ao))
       
        # loss =(1/100)*np.sum(-one_hot_labels * np.log(ao))+(1/200*np.sum(abs(wo)))  Regularized cross-entropy function
        
        print('Loss function value: ', loss)
        error_cost.append(loss)
        epoch+=1
        epochs.append(epoch)
        accuracy=(len(training_set)-loss)/len(training_set)
        Accuracy.append(accuracy)
        
        

#### Iniitializing the weights with final weight values

f_wh= wh
f_bh= bh
f_wo= wo
f_bo= bo

#########

def test_data():
    
          zh = np.dot(test_set, f_wh) + f_bh
          ah = sigmoid(zh)

          
          zo = np.dot(ah, f_wo) + f_bo
          ao1 = softmax(zo)
          loss=np.sum(-one_hot_labels_t * np.log(ao1))    
          #loss =(1/100)*np.sum(-one_hot_labels_t * np.log(ao))+(1/200*np.sum(abs(wo))) Regularized cross-entropy function
          accuracy=(len(test_set)-loss)/len(test_set)
          print('Accuracy:',accuracy)
          print('Loss function value: ', loss)
          Accuracy.append(accuracy)
          

          #error_cost.append(loss)
print("\n\nTesting phase")
test_data()

##### Plotting the outputs

plt.plot(epochs,error_cost)
plt.xlabel('Number of epochs')
plt.ylabel('loss')
plt.show()
epoch+=1
epochs.append(epoch)

plt.plot(epochs,Accuracy,color='green')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.show()
