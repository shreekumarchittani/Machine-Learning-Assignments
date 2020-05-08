
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from random import random
from math import exp #for sigmoid activation function


# In[9]:


#initializing the neuralNetwork for weights
def initNetwork(inputsNum, hiddenNum, outputsNum):
    mynetwork=list()
    hiddenLayer = [{'weights':[random() for i in range(inputsNum + 1)]} for i in range(hiddenNum)]
    mynetwork.append(hiddenLayer) #Hidden layer is the in between layer. Since user does not interact with that layer its called as hidden layer
    outputLayer = [{'weights':[random() for i in range(hiddenNum + 1)]} for i in range(outputsNum)]
    mynetwork.append(outputLayer)
    return mynetwork


# In[14]:


#activation function
def activate(weights, inputs):
    activationNum=weights[-1] # Last value
    for i in range(len(weights)-1):
        activationNum+=weights[i]*inputs[i]
    return activationNum

def sigmoidActivationFunction(activationNum):
    return 1.0 / (1.0 + exp(-activationNum))
    #This is the sigmoid activationNum function


# In[15]:


#forward propogation
def forwardProp(neuralNetwork,row):
    inputs=row
    for neuralLayer in neuralNetwork:
        new_inputs=[]
        for perceptron in neuralLayer:
            activationNum=activate(perceptron['weights'], inputs)
            perceptron['output']= sigmoidActivationFunction(activationNum)
            new_inputs.append(perceptron['output'])
        inputs=new_inputs
    return inputs




# In[16]:


#backward propogation to learn
def BPError(neuralNetwork, expected):
    for i in reversed(range(len(neuralNetwork))):
        neuralLayer = neuralNetwork[i]
        errors = list()
        if i != len(neuralNetwork)-1:
            for j in range(len(neuralLayer)):
                error = 0.0
                for perceptron in neuralNetwork[i + 1]:
                    error += (perceptron['weights'][j] * perceptron['delta'])
                errors.append(error)
        else:
            for j in range(len(neuralLayer)):
                perceptron = neuralLayer[j]
                errors.append(expected[j] - perceptron['output'])
        for j in range(len(neuralLayer)):
            perceptron = neuralLayer[j]
            perceptron['delta'] = errors[j] * (perceptron['output'] * (1.0 - perceptron['output']))


# In[17]:


#update weights on training
def update_weights(neuralNetwork, row, l_rate):
    for i in range(len(neuralNetwork)):
        inputs=row[:-1]
        if i!=0:
            inputs=[perceptron['output'] for perceptron in neuralNetwork[i-1]]
        for perceptron in neuralNetwork[i]:
            for j in range(len(inputs)):
                perceptron['weights'][j]+=l_rate*perceptron['delta']*inputs[j]
            perceptron['weights'][-1]+=l_rate*perceptron['delta']


# In[18]:


#training the neuralNetwork
def train(neuralNetwork, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forwardProp(neuralNetwork, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            BPError(neuralNetwork, expected)
            update_weights(neuralNetwork, row, l_rate)
        print('epoch=%d, lrate=%.5f, error=%.5f' % (epoch, l_rate, sum_error))


# In[19]:


#predicting function
def predict(neuralNetwork, row):
    outputs = forwardProp(neuralNetwork, row)
    return outputs.index(max(outputs))


# In[20]:
    
def main():
    samplesNum = 120 #total records
    featuresNum = 8 #total columns except outcome
    redundantNum = 1 #dependent variable
    classesNum = 2 # there aare two classes 0 and 1


# In[10]:
    data=pd.read_csv('/home/saddra/Desktop//newdataset.csv',index_col=0)
    data.head()

    X, y = make_classification(n_samples=samplesNum, n_features=featuresNum,n_redundant=redundantNum, n_classes=classesNum)
    data = pd.DataFrame(X, columns=['Pregnancies', 'Glucose', 'BloodPressure','BMI','DiabetesPedigreeFunction','hba1c','urine','weight'])

    data['label'] = y
    data.head()
    


    dataset=np.array(data[:])

    inputsNum = len(dataset[0]) - 1
    outputsNum = len(set([row[-1] for row in dataset]))
   
    train_dataset=dataset[:83]
    test_dataset=dataset[83:]


# In[23]:


#feeding the datset into the neuralNetwork
    neuralNetwork= initNetwork(inputsNum,1,outputsNum)
    train(neuralNetwork, train_dataset, 0.5, 100, outputsNum)


# In[24]:


#learned weights of the neuralNetwork
    for neuralLayer in neuralNetwork:
        print(neuralLayer)


# In[25]:


    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix


# In[26]:


#applying on training dataset
    y_train=[]
    pred=[]
    for row in train_dataset:
        prediction = predict(neuralNetwork, row)
        y_train.append(int(row[-1]))
        pred.append(prediction)


# In[27]:


    print("Accuracy: ",accuracy_score(y_train,pred))
    print("Confusion Matrix: ",confusion_matrix(y_train,pred))
    print("Precision: ",precision_score(y_train, pred))
    print("recall: ",recall_score(y_train, pred))


# In[28]:


#applying on testing dataset
    y_test=[]
    pred=[]
    for row in test_dataset:
        prediction = predict(neuralNetwork, row)
        y_test.append(row[-1])
        pred.append(prediction)


# In[29]:


    print("Accuracy: ",accuracy_score(y_test,pred))
    print("Confusion Matrix: ",confusion_matrix(y_test,pred))
    print("Precision: ",precision_score(y_test, pred))
    print("recall: ",recall_score(y_test, pred))

if __name__ == '__main__':
    main()
