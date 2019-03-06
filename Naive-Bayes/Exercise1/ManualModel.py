import os 
import csv
import time
import math
import random
import pandas
import numpy as np
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix

corpus_url = 'C:/Users/Aldrin/Desktop/school_folders/CS191-ML/trec07p/'
preprocessedCSV = 'preprocessed.csv'

def manual_NB(l, vocabTrain, vocabTest, targetTrain, targetTest,type):
  lamb = l

  ham_count = len(targetTrain[targetTrain==0])
  spam_count = len(targetTrain[targetTrain==1])
  total_count = ham_count + spam_count
  features = list(vocabTrain.columns.values)
  num_features = len(features)
  
  laplace_smoothing = lamb * 2
  
  p_ham = (ham_count + lamb) / (total_count + laplace_smoothing)
  p_spam = (spam_count + lamb) / (total_count + laplace_smoothing)
   
  X_ham = vocabTrain.loc[targetTrain[targetTrain==0].index]
  X_spam = vocabTrain.loc[targetTrain[targetTrain==1].index]  
  
  p_feature_is_1_given_ham = []
  p_feature_is_1_given_spam = []
  
  laplace_smoothing = lamb * num_features
  
  for f in features:
    num_of_ham_with_f = len(X_ham.loc[X_ham[f]==1])
    p_f_is_1_given_ham = (num_of_ham_with_f + lamb) / (ham_count + laplace_smoothing)
    p_feature_is_1_given_ham.append(p_f_is_1_given_ham)
    
    num_of_spam_with_f = len(X_spam.loc[X_spam[f]==1])
    p_f_is_1_given_spam = (num_of_spam_with_f + lamb) / (spam_count + laplace_smoothing)
    p_feature_is_1_given_spam.append(p_f_is_1_given_spam)
 
  prediction = []
  for i in range(0,len(vocabTest)):
    given = list(vocabTest.iloc[i].values)
    
    p_features_given_ham = 1
    p_features_given_spam = 1
   
    for j in range(0, num_features): 
      if(given[j]==1):
        p_features_given_ham*= p_feature_is_1_given_ham[j]
        p_features_given_spam*=p_feature_is_1_given_spam[j]
      else:
        p_features_given_ham*=(1-p_feature_is_1_given_ham[j])
        p_features_given_spam*=(1-p_feature_is_1_given_spam[j])
        
        
    p_ham_given_features = p_features_given_ham * p_ham
    p_spam_given_features = p_features_given_spam * p_spam
    
    if(p_ham_given_features > p_spam_given_features):
      prediction.append(0)
    else:
      prediction.append(1)
    
  target = list(targetTest.values)
  
  accuracy = 0
  for i in range(0,len(target)):
    if(target[i] == prediction[i]):
      accuracy+=1
  
  if(l!=0):
    print(type+" Vocabulary with Laplace Smoothing")
  else:
    print(type+" Vocabulary without Laplace Smoothing")
    
  accuracy = accuracy/len(prediction)
  print(accuracy)

def train():

  df_general = pandas.read_csv(os.path.join(corpus_url,preprocessedCSV))
  X_general = df_general.drop(['tokenDF$is_spam'],axis=1)
  y_general = df_general['tokenDF$is_spam']
  
  # df_reduced = pandas.read_csv('ham-spam-dataset-reduced.csv')
  # X_reduced = df_reduced.drop(['case_decision','target'],axis=1)
  # y_reduced = df_reduced['target']
  
  vocabTrain_general, vocabTest_general, targetTrain_general, targetTest_general = train_test_split(X_general, y_general, test_size=0.30)
  
  # vocabTrain_reduced = X_reduced.loc[vocabTrain_general.index]
  # vocabTest_reduced = X_reduced.loc[vocabTest_general.index]
  # targetTrain_reduced = y_reduced.loc[targetTrain_general.index]
  # targetTest_reduced = y_reduced.loc[targetTest_general.index]
  
  manual_NB(0,vocabTrain_general, vocabTest_general, targetTrain_general, targetTest_general ,"General")
  #manual_NB(0,vocabTrain_reduced, vocabTest_reduced, targetTrain_reduced, targetTest_reduced, "Reduced")
  manual_NB(1,vocabTrain_general, vocabTest_general, targetTrain_general, targetTest_general ,"General")
  #manual_NB(1,vocabTrain_reduced, vocabTest_reduced, targetTrain_reduced, targetTest_reduced, "Reduced")
  
  #sk_learn_NB(vocabTrain_general, vocabTest_general, targetTrain_general, targetTest_general ,"General")
  #sk_learn_NB(vocabTrain_reduced, vocabTest_reduced, targetTrain_reduced, targetTest_reduced, "Reduced")
  
def main():
  start_time = datetime.now().time().strftime('%H:%M:%S')
    
  train()
  
  end_time = datetime.now().time().strftime('%H:%M:%S')
  total_time=(datetime.strptime(end_time,'%H:%M:%S') - datetime.strptime(start_time,'%H:%M:%S'))
  print("Training: ",total_time)
  #buffer = input("...DONE...")
  #print(buffer)
 
if __name__ == '__main__':
  main()
  #buffer = input("")