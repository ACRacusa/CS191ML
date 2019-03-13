import email, re, os, string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')
path = 'C:/Users/Aldrin/Desktop/school_folders/CS191-ML/trec07p/'
tokenizedData = 'tokenizedData.csv'
csvResult = 'svmResult.csv'

def pipelineClassifier(msg_train,msg_test,label_train,label_test,svm_gamma,svm_cost,svm_kernel):
    print("with training and testing with Grid Search C={:f}, kernel={}, gamma={}".format(svm_cost,svm_kernel,svm_gamma))
    #print('Training set={:d}, Test Set={:d}'.format(len(msg_train),len(msg_test)))
    pipeline = Pipeline([
        ('bow',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('classifier',SVC(C=svm_cost,kernel=svm_kernel,gamma=svm_gamma)),
        ])

    pipeline.fit(msg_train,label_train)
    predictions = pipeline.predict(msg_test)
    #print(classification_report(predictions,label_test))
    #print(confusion_matrix(label_test,predictions))
    pipelineAccuracy = accuracy_score(label_test,predictions)
    pipelinePrecision = precision_score(label_test,predictions,average="weighted")
    pipelineRecall = recall_score(label_test,predictions,average="weighted")
    # print(precision_score(label_test,predictions,average="weighted"))
    # print(recall_score(label_test,predictions,average="weighted"))
    # print(accuracy_score(label_test,predictions))
    return [pipelineAccuracy,pipelinePrecision,pipelineRecall,svm_gamma,svm_kernel,svm_cost]
    

if __name__ == "__main__":
    spamClassifier = pd.read_csv(os.path.join(path, tokenizedData), usecols=['is_spam', 'text'])
    spamClassifier = spamClassifier.iloc[0:1000]
    #print(spamClassifier.groupby('is_spam').count())

    #replace '' values with NAN
    spamClassifier['text'].replace('', np.nan, inplace=True)
    #drop NAN values
    spamClassifier.dropna(how='any', inplace=True)

    #split test and training
    msg_train,msg_test,label_train,label_test = train_test_split(spamClassifier['text'],spamClassifier['is_spam'],test_size=0.25)
    
    gamma_params = [1,2,3,4,'auto']
    c_params = [1.0,0.75,0.5,0.25]
    kernel_params = ['linear','rbf','poly','sigmoid']
    df = pd.DataFrame(columns=['accuracy', 'precision', 'recall','gamma','kernel','cost'])    
    counter = 0
    for i in gamma_params:
        for j in c_params:
            for k in kernel_params:
                df.loc[counter] = pipelineClassifier(msg_train,msg_test,label_train,label_test,i,j,k)
                counter +=1
    print(df)

    df.to_csv(os.path.join(path,csvResult))
    