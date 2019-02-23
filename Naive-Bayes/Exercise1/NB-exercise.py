import email, re, os, threading, string
import pandas as pd
import numpy as np
import mailparser
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

stemming = PorterStemmer()
stops = set(stopwords.words("english")) 
path = 'C:/Users/Aldrin/Desktop/school_folders/CS191-ML/trec07p/'
indexFilePath = 'full/index'
csvFileName = 'parsedEmails.csv'
tokenizedData = 'tokenizedData.csv'
#Source: https://www.kaggle.com/dilip990/spam-ham-detection-using-naive-bayes-classifier
def text_process(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def cleanhtml(raw_html):
    #cleanr = re.compile('<[^<]+?>')
    #cleantext = re.sub(cleanr, '', raw_html)
    #removes the unnecessary unicode characters
    #cleantext = raw_html.encode('utf-8', 'ignore')
    cleantext = re.sub(r'\\[Uu][a-zA-Z0-9]{4}', '', raw_html)
    #cleantext = BeautifulSoup(cleantext) 
    cleantext = cleanMe(cleantext)
    return cleantext

#Source: https://stackoverflow.com/questions/30565404/remove-all-style-scripts-and-html-tags-from-an-html-page/30565420
def cleanMe(html):
    soup = BeautifulSoup(html,"html.parser") # create a new bs4 object from the html data loaded
    for script in soup(["script", "style"]): # remove all javascript and stylesheet code
        script.extract()
    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = ' '.join(chunk for chunk in chunks if chunk)
    return text.replace('\t','')

#Source: https://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space
# and for the ord(): https://www.programiz.com/python-programming/methods/built-in/ord
def deleteNonASCII(text):
    return ''.join([i if ord(i) < 128 else '' for i in text])

def deprecatedPreprocess(spamClassifier):
   bow_transformer = CountVectorizer(analyzer=text_process).fit(spamClassifier['text'])
   print('bow length: ',len(bow_transformer.vocabulary_))
   message4=spamClassifier['text'][3]
   bow4=bow_transformer.transform([message4])
   print('bow4 ',bow4)
   print('bow4 shape: ',bow4.shape)

   messages_bow = bow_transformer.transform(spamClassifier['text'])
   print('Shape of Sparse Matrix: ',messages_bow.shape)
   print('Amount of non-zero occurences:',messages_bow.nnz)

   tfidf_transformer=TfidfTransformer().fit(messages_bow)
   tfidf4 = tfidf_transformer.transform(bow4)
   print(tfidf4)

   messages_tfidf=tfidf_transformer.transform(messages_bow)
   print(messages_tfidf.shape)

   spam_detect_model = MultinomialNB().fit(messages_tfidf,spamClassifier['text'])
   print('predicted:',spam_detect_model.predict(tfidf4)[0])
   print('expected:',spamClassifier.text[3])

def stem_list(row):
    my_list = row
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)
def remove_stops(row):
    my_list = row
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)

def rejoin_words(row):
    my_list = row
    joined_words = ( " ".join(my_list))
    return joined_words
if __name__ == "__main__":
   spamClassifier = pd.read_csv(os.path.join(path, csvFileName), usecols=['is_spam', 'text'])
   spamClassifier['text'] = spamClassifier['text'].astype(str)
   #spamClassifier = spamClassifier.iloc[0:5]
   spamClassifier['text'] = spamClassifier['text'].apply(deleteNonASCII)
   spamClassifier['text'] = spamClassifier['text'].apply(cleanhtml)
   #spamClassifier['text'].replace('', np.nan, inplace=True)
   #spamClassifier['text'].dropna(how='any', inplace=True)
   #lowercase
   #spamClassifier['text'] = spamClassifier['text'].map(lambda x: x.lower()) 
   spamClassifier['text'] = spamClassifier['text'].str.lower()
   #remove punctuations
   spamClassifier['text'] = spamClassifier['text'].str.replace('[^\w\s]', '') 
   #replace '' values with NAN
   spamClassifier['text'].replace('', np.nan, inplace=True)
   #drop NAN values
   spamClassifier['text'].dropna(how='any', inplace=True)

   # tokenize the data and remove stopwords 
   spamClassifier['text'] = spamClassifier['text'].apply(text_process)
   #spamClassifier['text'] = spamClassifier['text'].apply(nltk.word_tokenize) 
#    #apply stemming
#    spamClassifier['text'] = spamClassifier['text'].apply(stem_list)

#    #remove stopwords
#    spamClassifier['text'] = spamClassifier['text'].apply(remove_stops)

#    spamClassifier['text'] = spamClassifier['text'].apply(rejoin_words)
   #nltk.download()
   #stemmer = PorterStemmer()
   #spamClassifier['text'] = spamClassifier['text'].apply(lambda x: [stemmer.stem(y) for y in x])  
   
   
   #print(spamClassifier)
   spamClassifier.to_csv(os.path.join(path, tokenizedData))
   #print(spamClassifier.groupby('text').describe())
   #Preprocess?

   