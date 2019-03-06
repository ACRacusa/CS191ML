import email, re, os, string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from collections import defaultdict

#stops = set(stopwords.words("english")) 
path = 'C:/Users/Aldrin/Desktop/school_folders/CS191-ML/trec07p/'
tokenizedData = 'tokenizedData.csv'


if __name__ == "__main__":
    spamClassifier = pd.read_csv(os.path.join(path, tokenizedData), usecols=['is_spam', 'text'])
    #spamClassifier = spamClassifier.iloc[0:100]
    print(spamClassifier.groupby('is_spam').count())
    proby_spam = 50199
    proby_ham = 25220
    msg_train,msg_test,label_train,label_test = train_test_split(spamClassifier['text'],spamClassifier['is_spam'],test_size=0.25)

    #replace '' values with NAN
    spamClassifier['text'].replace('', np.nan, inplace=True)
    #drop NAN values
    spamClassifier['text'].dropna(how='any', inplace=True)
    #bow_transformer = CountVectorizer(analyzer=text_process).fit(spamClassifier['text'])
    bow_transformer = CountVectorizer().fit(spamClassifier['text'])
    #print(len(bow_transformer.vocabulary_))

    messages_bow = bow_transformer.transform(spamClassifier['text'])
    #print('Shape of Sparse Matrix: ',messages_bow.shape)
    #print('Amount of non-zero occurences:',messages_bow.nnz)
    sparsity =(100.0 * messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1]))
    #print('sparsity:{}'.format(round(sparsity)))

    tfidf_transformer=TfidfTransformer().fit(messages_bow)

    messages_tfidf=tfidf_transformer.transform(messages_bow)
    #print(messages_tfidf.shape) 
    print(bow_transformer.get_feature_names())
    print(messages_tfidf.toarray())

    # #training a model

    # spam_detect_model = MultinomialNB().fit(messages_tfidf,spamClassifier['is_spam'])

    # all_predictions = spam_detect_model.predict(messages_tfidf)
    # #print(all_predictions)

    # #print(classification_report(spamClassifier['is_spam'],all_predictions))
    # #print(confusion_matrix(spamClassifier['is_spam'],all_predictions))

    # #split test and training
    # msg_train,msg_test,label_train,label_test = train_test_split(spamClassifier['text'],spamClassifier['is_spam'],test_size=0.25)

    # print(len(msg_train),len(msg_test),len(label_train),len(label_test))

    # ##no laplace smoothing
    # print("without laplace smoothing General Vocabulary")
    # pipeline = Pipeline([
    #     ( 'bow',CountVectorizer()),
    #     ('tfidf',TfidfTransformer()),
    #     ('classifier',MultinomialNB(alpha=1.0e-10)),
    #     ])

    # pipeline.fit(msg_train,label_train)
    # predictions = pipeline.predict(msg_test)
    # print(classification_report(predictions,label_test))
    # print(confusion_matrix(label_test,predictions))
    # #skplt.metrics.plot_confusion_matrix(label_test, predictions, normalize=True)
    # #plt.show()
    # #plot_classification_report(classification_report(predictions,label_test), with_avg_total=True)
    # ##### with laplace smoothing
    # print("with laplace smoothing General Vocabulary")
    # pipelineLaplaceGen = Pipeline([
    #     ('bow', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('classifier', MultinomialNB())
    # ])

    # pipelineLaplaceGen.fit(msg_train,label_train)
    # predictionsLaplaceGenVocab = pipelineLaplaceGen.predict(msg_test)
    # print(classification_report(predictionsLaplaceGenVocab,label_test))
    # print(confusion_matrix(label_test,predictionsLaplaceGenVocab))
    # #plot_classification_report(classification_report(predictionsLaplaceGenVocab,label_test), with_avg_total=True)

    # #source: https://stackoverflow.com/questions/51695769/sklearn-chi2-for-feature-selection
    # ###### without Laplace Smoothing reduced words
    # print("Without Laplace Smoothing Reduced words (200)")
    # pipelineNoLaplaceReduced = Pipeline([
    #     ('bow', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('reduce', SelectKBest(chi2, k=200)),
    #     ('classifier', MultinomialNB(alpha=1.0e-10))
    # ])

    # pipelineNoLaplaceReduced.fit(msg_train,label_train)
    # predictionsNoLaplaceReduced = pipelineNoLaplaceReduced.predict(msg_test)
    # print(classification_report(predictionsNoLaplaceReduced,label_test))
    # print(confusion_matrix(label_test,predictionsNoLaplaceReduced))
    # #plot_classification_report(classification_report(predictionsNoLaplaceReduced,label_test), with_avg_total=True)
    # #### With laplace Smoothing reduced words
    # print("With Laplace Smoothing Reduced words (200)")
    # pipelineLaplaceReduced = Pipeline([
    #     ('bow', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('reduce', SelectKBest(chi2, k=200)),
    #     ('classifier', MultinomialNB())
    # ])

    # pipelineLaplaceReduced.fit(msg_train,label_train)
    # predictionsLaplaceReduced = pipelineLaplaceReduced.predict(msg_test)
    # print(classification_report(predictionsLaplaceReduced,label_test))
    # print(confusion_matrix(label_test,predictionsLaplaceReduced))
    # #plot_classification_report(classification_report(predictionsLaplaceReduced,label_test), with_avg_total=True)







