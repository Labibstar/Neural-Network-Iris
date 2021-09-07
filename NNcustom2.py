# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:11:31 2021

@author: Labib Rahman & MD Zafore Sadek
"""

import numpy as np
import matplotlib.pyplot as plt



#### Importing the Data ####

path=r'C:\Users\USER\Downloads\Iris.csv'

numpy_array = np.genfromtxt(path,delimiter=",", skip_header=1,encoding="utf8",dtype=np.str, usecols=(1,2,3,4))


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
one_hot_labels_s = np.zeros((1,3))

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

instances = feature_set.shape[0]
attributes = feature_set.shape[1]
hidden_nodes = 3
output_labels = 3

wa=np.random.rand(attributes,hidden_nodes)
wm = np.random.rand(hidden_nodes,hidden_nodes)


ba = np.random.randn(hidden_nodes)
bm= np.random.randn(hidden_nodes)



wo = np.random.rand(hidden_nodes,output_labels)
bo = np.random.randn(output_labels)
lr = 0.0001

error_cost = []
epoch=0
epochs=[]
Accuracy=[]
AO=[]
iterations=0
Iterations=[]

########## Training Phase ###########
print("Testing Phase \n")
for iterations in range(1000):
 for epoch in range(100):

############# feedforward
    

   ## Creating stochastic set
   
    i=np.random.choice(100)
    stochastic_set=training_set[i]
    stochastic_set=stochastic_set.reshape(1,4)    
    one_hot_labels_s=training_labels[i]
    
    
    # Phase 1
    za = np.dot(stochastic_set, wa) + ba
    ah = sigmoid(za)
    #print(ah.shape)
    #print(wm.shape)
        
    zm = ah.dot(wm) + bm
    mh = sigmoid(zm)
    

    # Phase 2
    zo = np.dot(mh,wo) + bo
    ao = softmax(zo)
    AO.append(ao)
    
########## Back Propagation

########## Phase 1

    dL_dzo = ao - one_hot_labels_s
    
    dzo_dwo = mh

    dL_dwo = np.dot(dzo_dwo.T, dL_dzo)

    dL_dbo = dL_dzo

########## Phase 2
    dzo_dmh = wo # 8*3
    
       
    dzm_dwm= ah
    dzo_dzm = sigmoid_der(mh)
    
    dL_dwmA=np.dot(dL_dzo,dzo_dmh.T)
   
    dL_dwmB=np.multiply(dL_dwmA,dzo_dzm)

        
    dL_dwm=np.dot(ah.T,dL_dwmB)
    #print(dL_dwm.shape)
    
    
    dL_dmh=np.dot(dL_dzo, dzo_dmh.T)
    dL_dbm =np.multiply(dL_dmh,dzo_dzm)
    
######### Phase 3
   
    dzm_dza=sigmoid_der(ah)
    
    dza_dwa = stochastic_set
    
    dzm_dah =wm
    
    dL_dmh= np.dot(dL_dzo, dzo_dmh.T)
    dmh_dah=np.dot(dzo_dzm,dzm_dah.T)
    dmh_dwa=np.multiply(dL_dmh,dmh_dah)
   
    dL_dah=np.multiply(dmh_dwa,dzm_dza)
    
    dL_dwa=np.dot(stochastic_set.T,dL_dah)

    
    
   
    
    dL_dmh1=np.dot(dL_dzo,dzo_dmh.T)
    dL_dah1=np.multiply(dL_dmh1,np.dot(dzo_dzm,dzm_dah))
    dL_dba = np.multiply(dL_dah1,dzm_dza)
    
    # Update Weights ================
    
    wa -= lr * dL_dwa
    ba -= lr * dL_dba.sum(axis=0)
    
    wm -= lr * dL_dwm
    bm -= lr * dL_dbm.sum(axis=0)

    wo -= lr * dL_dwo
    bo -= lr * dL_dbo.sum(axis=0)
    
   
 
 iterations+=1
 Iterations.append(iterations) 
 if iterations % 1 == 0:
        Output=np.array(AO) 
        Output=Output.reshape(100,3)
        loss =(1/100)*np.sum(-one_hot_labels * np.log(Output))+(1/200*np.sum(abs(Output)))
        error_cost.append(loss)
        epoch+=1
        epochs.append(epoch)
        accuracy=(len(training_set)-loss)/len(training_set)
        Accuracy.append(accuracy)
        AO.clear()
        if iterations % 100 ==0:
           print('Loss function value: ', loss)
        
        
error_cost.pop()


#### Iniitializing the weights with final weight values

f_wa= wa
f_ba= ba
f_wm= wm
f_bm= bm

f_wo= wo
f_bo= bo

################## Testing Phase #################

def test_data():
    
                    # Phase 1
            za = np.dot(test_set, wa) + ba
            ah = sigmoid(za)
           
                
            zm = ah.dot(wm) + bm
            mh = sigmoid(zm)
            
        
                    # Phase 2
            zo = np.dot(mh,wo) + bo
            ao = softmax(zo)

            loss=np.sum(-one_hot_labels_t * np.log(ao))    
            #loss =(1/50)*np.sum(-one_hot_labels_t * np.log(ao))+(1/100*np.sum(abs(wo)))
            print('Loss function value: ', loss)
            
            error_cost.append(loss)
            accuracy=(len(test_set)-loss)/len(test_set)*100
            print('Accuracy:',accuracy)
            Accuracy.append(accuracy)
            
print("\n\nTesting phase")
test_data()

####### Plotting Output ########

plt.plot(Iterations,error_cost)
plt.xlabel('Number of epochs')
plt.ylabel('loss')
plt.show()
iterations+=1
Iterations.append(iterations)
plt.plot(Iterations,Accuracy,color='green')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.show()