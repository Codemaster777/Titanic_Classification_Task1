#NAME:BHANUSHRAY GUPTA
#DATA SCIENCE TASK 1
#BHARAT INTERN
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import seaborn as sns
import re
from DATA_PROCESSING import preProcess

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Machine Learning Algorithmns
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def sType(str):
  if (str == '1'):
    return "Survived"
  else:
    return "Deceased"


def printPrediction(predList):
  i = 1
  s = ""
  for cell in predList:
    s = s + (str(i) + ": " +sType(str(cell))+ '\n')
    i = i + 1
  
  return s




global predLog 
def logRegression(train1,train2,test):
  model = LogisticRegression()
  model.fit(train1, train2)
  prediction = model.predict(test)
  global predLog
  predLog = prediction
  accuracy = round(model.score(train1,train2)*100,2)
  return 'PREDICTION PROCESSING:  \n \n'+ printPrediction(prediction) + '\nModel Accuracy: ' + str(accuracy)+'%'

 

global predK
def KNN(train1,train2,test):
  model = KNeighborsClassifier(n_neighbors=3)
  model.fit(train1,train2)
  prediction = model.predict(test)
  global predK
  predK = prediction
  print(prediction)
  accuracy = round(model.score(train1,train2)*100,2)
  return 'PREDICTION PROCESSING:\n \n'+ printPrediction(prediction) + '\nModel Accuracy: ' + str(accuracy)+'%'



global predBayes
def gNaiveBayes(train1,train2,test):
  model = GaussianNB()
  model.fit(train1,train2)
  prediction = model.predict(test)
  global predBayes
  predBayes = prediction
  accuracy = round(model.score(train1,train2)*100,2)
  return 'PREDICTION PROCESSING: \n \n'+ printPrediction(prediction) + '\nModel Accuracy: ' + str(accuracy)+'%'




global predTree
def dTree(train1,train2,test):
  model = DecisionTreeClassifier()
  model.fit(train1,train2)
  prediction = model.predict(test)
  global predTree
  predTree = prediction
  accuracy = round(model.score(train1,train2)*100,2)
  return 'PREDICTION PROCESSING: \n \n'+ printPrediction(prediction) + '\nModel Accuracy: ' + str(accuracy)+'%'

def trainclassDistr(train_data):
    train_data.Pclass.value_counts().plot(kind="barh")
    plt.title("CLASS DISTRIBUTION")
    plt.xlabel("NUMBER OF PASSENGERS")
    plt.ylabel("TICKET CLASS TYPE")
    plt.legend(["1st (Upper), 2nd (Middle), 3rd (Lower)"])
    plt.show()

def trainMeanFareSurvival(train_data):
    survived_0 = train_data[train_data['Survived'] == 0]["Fare"].mean()
    survived_1 = train_data[train_data['Survived'] == 1]["Fare"].mean()
    xs  = [survived_0, survived_1]
    ys = ['Dead','Survived']
    plt.bar(ys, xs, 0.6, align='center',color = 'green')
    plt.ylabel('AVERAGE FARE')
    plt.title('AVERAGE FARE VS SURVIVAL')
    plt.xlabel('OUTCOMES')
    plt.show()

def trainClassSurvival(train_data):
    pclass_survived = train_data[train_data['Survived']==1]['Pclass'].value_counts()
    pclass_dead = train_data[train_data['Survived']==0]['Pclass'].value_counts()
    df = pd.DataFrame([pclass_survived,pclass_dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,8))
    plt.ylabel('NUMBER OF PASSENGERS')
    plt.legend(["1st (Upper)", "2nd (Middle)","3rd (Lower)"])
    plt.title('TICKET CLASS VS SURVIVAL')
    plt.show()

def groupPlot(data):
    status = ('Deceased', 'Survived')
    y_pos = np.arange(len(status))
    count1 = 0
    for e in data:
        if (e == 1):
            count1 = count1 + 1
        
    count0 = 0
    for z in data:
        if (z == 0):
            count0 = count0 + 1                                                                 
            
    predict = [count0,count1]                                                            
    np.sort(data)
    plt.bar(y_pos, predict, align='center', color='blue', alpha=0.5)
    plt.xticks(y_pos, status)
    plt.ylabel('NUMBER OF PASSENGERS')
    plt.title('SURVIVED VS DECEASED')
    plt.show()

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
processedData = preProcess(train_df)
processedTestData = preProcess(test_df)

train1 = processedData.drop("Survived", axis = 1)
train2 = processedData["Survived"]
test = processedTestData.copy()

