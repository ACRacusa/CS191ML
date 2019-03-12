import email, re, os, string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')
path = 'C:/Users/Aldrin/Desktop/school_folders/CS191-ML/trec07p/'
tokenizedData = 'tokenizedData.csv'


if __name__ == "__main__":
    spamClassifier = pd.read_csv(os.path.join(path, tokenizedData), usecols=['is_spam', 'text'])
    #spamClassifier = spamClassifier.iloc[0:1000]
    #print(spamClassifier.groupby('is_spam').count())

    #replace '' values with NAN
    spamClassifier['text'].replace('', np.nan, inplace=True)
    #drop NAN values
    spamClassifier.dropna(how='any', inplace=True)
    #bow_transformer = CountVectorizer(analyzer=text_process).fit(spamClassifier['text'])
    #print(spamClassifier)
    #split test and training
    msg_train,msg_test,label_train,label_test = train_test_split(spamClassifier['text'],spamClassifier['is_spam'],test_size=0.25)

    # bow_transformer = CountVectorizer().fit(spamClassifier['text'])

    # messages_bow = bow_transformer.transform(spamClassifier['text'])

    # tfidf_transformer=TfidfTransformer().fit(messages_bow)

    # messages_tfidf=tfidf_transformer.transform(messages_bow)

    # # #training a model

    # spam_detect_model = SVC(C=1,kernel='linear',gamma='auto').fit(messages_tfidf,spamClassifier['is_spam'])

    # all_predictions = spam_detect_model.predict(messages_tfidf)
    # # #print(all_predictions)

    # print(classification_report(spamClassifier['is_spam'],all_predictions))
    # print(confusion_matrix(spamClassifier['is_spam'],all_predictions))
    
    # print("with training and testing without Grid Search,linear,c=1")
    # print('Training set={:d}, Test Set={:d}'.format(len(msg_train),len(msg_test)))
    # pipeline = Pipeline([
    #     ( 'bow',CountVectorizer()),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',SVC(C=1,kernel='linear',gamma='auto')),
    #     ])

    # pipeline.fit(msg_train,label_train)
    # predictions = pipeline.predict(msg_test)
    # print(classification_report(predictions,label_test))
    # print(confusion_matrix(label_test,predictions))

    # print("with training and testing without Grid Search poly, c=1")
    # print('Training set={:d}, Test Set={:d}'.format(len(msg_train),len(msg_test)))
    # pipeline = Pipeline([
    #     ( 'bow',CountVectorizer()),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',SVC(C=2,kernel='poly',gamma='auto')),
    #     ])

    # pipeline.fit(msg_train,label_train)
    # predictions = pipeline.predict(msg_test)
    # print(classification_report(predictions,label_test))
    # print(confusion_matrix(label_test,predictions))

    # print("with training and testing without Grid Search sigmoid, c=1")
    # print('Training set={:d}, Test Set={:d}'.format(len(msg_train),len(msg_test)))
    # pipeline = Pipeline([
    #     ( 'bow',CountVectorizer()),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',SVC(C=1,kernel='sigmoid',gamma='auto')),
    #     ])

    # pipeline.fit(msg_train,label_train)
    # predictions = pipeline.predict(msg_test)
    # print(classification_report(predictions,label_test))
    # print(confusion_matrix(label_test,predictions))

    # print("with training and testing without Grid Search rbf, c=1")
    # print('Training set={:d}, Test Set={:d}'.format(len(msg_train),len(msg_test)))
    # pipeline = Pipeline([
    #     ( 'bow',CountVectorizer()),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',SVC(C=1,kernel='rbf',gamma='auto')),
    #     ])

    # pipeline.fit(msg_train,label_train)
    # predictions = pipeline.predict(msg_test)
    # print(classification_report(predictions,label_test))
    # print(confusion_matrix(label_test,predictions))


    # print("with training and testing without Grid Search,linear,c=2")
    # print('Training set={:d}, Test Set={:d}'.format(len(msg_train),len(msg_test)))
    # pipeline = Pipeline([
    #     ( 'bow',CountVectorizer()),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',SVC(C=2,kernel='linear',gamma='auto')),
    #     ])

    # pipeline.fit(msg_train,label_train)
    # predictions = pipeline.predict(msg_test)
    # print(classification_report(predictions,label_test))
    # print(confusion_matrix(label_test,predictions))

    # print("with training and testing without Grid Search poly, c=2")
    # print('Training set={:d}, Test Set={:d}'.format(len(msg_train),len(msg_test)))
    # pipeline = Pipeline([
    #     ( 'bow',CountVectorizer()),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',SVC(C=2,kernel='poly',gamma='auto')),
    #     ])

    # pipeline.fit(msg_train,label_train)
    # predictions = pipeline.predict(msg_test)
    # print(classification_report(predictions,label_test))
    # print(confusion_matrix(label_test,predictions))

    # print("with training and testing without Grid Search sigmoid, c=2")
    # print('Training set={:d}, Test Set={:d}'.format(len(msg_train),len(msg_test)))
    # pipeline = Pipeline([
    #     ( 'bow',CountVectorizer()),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',SVC(C=2,kernel='sigmoid',gamma='auto')),
    #     ])

    # pipeline.fit(msg_train,label_train)
    # predictions = pipeline.predict(msg_test)
    # print(classification_report(predictions,label_test))
    # print(confusion_matrix(label_test,predictions))

    print("with training and testing without Grid Search")
    print('Training set={:d}, Test Set={:d}'.format(len(msg_train),len(msg_test)))
    pipeline = Pipeline([
        ('bow',CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('classifier',SVC(C=1,kernel='linear',gamma='auto')),
        ])

    pipeline.fit(msg_train,label_train)
    predictions = pipeline.predict(msg_test)
    print(classification_report(predictions,label_test))
    print(confusion_matrix(label_test,predictions))

    # print("with training and testing\n")
    
    # bow_transformer = CountVectorizer().fit(msg_train)
    # messages_bow = bow_transformer.transform(msg_train)
    # tfidf_transformer=TfidfTransformer().fit(messages_bow)
    # messages_tfidf=tfidf_transformer.transform(messages_bow)
    # svmModel = SVC(C=1,kernel='linear',gamma='auto').fit(messages_tfidf,label_train)

    # predictions = svmModel.predict(messages_tfidf)
    # print(classification_report(predictions,label_test))
    # print(confusion_matrix(label_test,predictions))
    
    print("######################### with grid search#################################")

    spamClassifier = spamClassifier.iloc[0:2000]
    #print(spamClassifier.groupby('is_spam').count())

    #replace '' values with NAN
    spamClassifier['text'].replace('', np.nan, inplace=True)
    #drop NAN values
    spamClassifier.dropna(how='any', inplace=True)

    #split test and training
    msg_train,msg_test,label_train,label_test = train_test_split(spamClassifier['text'],spamClassifier['is_spam'],test_size=0.25)
    # Set the parameters by cross-validation
    bow_transformer = CountVectorizer().fit(msg_train)

    messages_bow = bow_transformer.transform(msg_train)

    tfidf_transformer=TfidfTransformer().fit(messages_bow)

    messages_tfidf=tfidf_transformer.transform(messages_bow)

    ##for test set
    bow_transformer_test = CountVectorizer().fit(msg_test)

    messages_bow_test = bow_transformer.transform(msg_test)

    tfidf_transformer_test=TfidfTransformer().fit(messages_bow_test)

    messages_tfidf_test=tfidf_transformer.transform(messages_bow_test)

    tuned_parameters = [{'kernel': ['rbf','linear','sigmoid','poly'], 'gamma': [1e-2,1e-3, 1e-4,'auto'],
                        'C': [1, 2, 5, 10, 100]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                        scoring='%s_macro' % score,n_jobs=-1)
        clf.fit(messages_tfidf, label_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = label_test, clf.predict(messages_tfidf_test)
        print(confusion_matrix(y_true,y_pred))
        print(classification_report(y_true, y_pred))
        print()

    # print("with training and testing with Grid Search")
    # print('Training set={:d}, Test Set={:d}'.format(len(msg_train),len(msg_test)))
    # pipeline = Pipeline([
    #     ( 'bow',CountVectorizer()),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',SVC(),
    #     ])

    # parameters = {'kernel':('linear', 'rbf','poly','sigmoid'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto')}
    # clf = GridSearchCV(pipeline, parameters, n_jobs=-1)
    # clf.fit(messages_tfidf,label_train)

    #pipeline.fit(msg_train,label_train)
    #predictions = pipeline.predict(msg_test)
    #print(classification_report(predictions,label_test))
    #print(confusion_matrix(label_test,predictions))

    # bow_transformer = CountVectorizer().fit(msg_train)

    # messages_bow = bow_transformer.transform(msg_train)

    # tfidf_transformer=TfidfTransformer().fit(messages_bow)

    # messages_tfidf=tfidf_transformer.transform(messages_bow)
    
    # svm = SVC()
    # parameters = {'kernel':('linear', 'rbf','poly','sigmoid'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto')}
    # clf = GridSearchCV(svm, parameters)
    # clf.fit(messages_tfidf,label_train)

    # bow_transformer = CountVectorizer().fit(msg_test)
    # messages_bow = bow_transformer.transform(msg_test)
    # tfidf_transformer=TfidfTransformer().fit(messages_bow)
    # messages_tfidf=tfidf_transformer.transform(messages_bow)
    
    # predictions = clf.predict(messages_tfidf)
    # print(classification_report(predictions,label_test))
    # print(confusion_matrix(label_test,predictions))

    # print(clf.best_params_)
    # print(clf.best_score_)

    # svmModel = SVC(C=1,kernel='linear',gamma='auto').fit(messages_tfidf,label_train)

    # predictions = svmModel.predict(messages_tfidf)
    # print(classification_report(predictions,label_test))
    # print(confusion_matrix(label_test,predictions))

    #all_predictions = clf.predict(msg_test)
    # print("accuracy:"+str(np.average(cross_val_score(clf, messages_tfidf, spamClassifier['is_spam'], scoring='accuracy'))))
    # print("f1:"+str(np.average(cross_val_score(clf, messages_tfidf, spamClassifier['is_spam'], scoring='f1'))))

    # all_predictions = clf.predict(msg_test)
    # # # #print(all_predictions)

    #print(classification_report(all_predictions,label_test))
    #print(confusion_matrix(all_predictions,label_test))






