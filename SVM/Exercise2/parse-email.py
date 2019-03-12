import email, re, os, threading,string
import pandas as pd
import numpy as np
import mailparser
from bs4 import BeautifulSoup

path = 'C:/Users/Aldrin/Desktop/school_folders/CS191-ML/trec07p/'
indexFilePath = 'full/index'
csvFileName = 'parsedEmails.csv'
exitFlag = 0

#parallel read email
def readEmail(thread_name, email_frame):
    for index, row in email_frame.iterrows():
        print(row['email_filename'])
        try:
            email_frame.at[index, 'text'] = parseEmailFromFile(os.path.join(path, row['email_filename']))
        except:
            email_frame.at[index, 'text'] = ''
            continue
    
    if exitFlag:
        thread_name.exit()
 

# def cleanhtml(raw_html):
#     #cleanr = re.compile('<[^<]+?>')
#     #cleantext = re.sub(cleanr, '', raw_html)
#     #removes the unnecessary unicode characters
#     #cleantext = raw_html.encode('utf-8', 'ignore')
#     cleantext = re.sub(r'\\[Uu][a-zA-Z0-9]{4}', '', raw_html)
#     #cleantext = BeautifulSoup(cleantext) 
#     cleantext = cleanMe(cleantext)
#     return cleantext

# #Source: https://stackoverflow.com/questions/30565404/remove-all-style-scripts-and-html-tags-from-an-html-page/30565420
# def cleanMe(html):
#     soup = BeautifulSoup(html,"html.parser") # create a new bs4 object from the html data loaded
#     for script in soup(["script", "style"]): # remove all javascript and stylesheet code
#         script.extract()
#     # get text
#     text = soup.get_text()
#     # break into lines and remove leading and trailing space on each
#     lines = (line.strip() for line in text.splitlines())
#     # break multi-headlines into a line each
#     chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
#     # drop blank lines
#     text = ' '.join(chunk for chunk in chunks if chunk)
#     return text.replace('\t','')

# #Source: https://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space
# # and for the ord(): https://www.programiz.com/python-programming/methods/built-in/ord
# def deleteNonASCII(text):
#     return ''.join([i if ord(i) < 128 else '' for i in text])

#Extract the emailfile from the given filename and then cleans the html tags
def parseEmailFromFile(emailFile):
    mail = mailparser.parse_from_file(emailFile)
    #noAsciiEmailBody = deleteNonASCII(mail.body)
    #email_body = cleanhtml(noAsciiEmailBody)
    #email_body.replace('', np.nan, inplace=True)
    #finalCleanEmail = email_body.replace('\n',' ').strip()
    return mail.body
    #return finalCleanEmail
    
#Preprocess the "/full/index"
def preprocessDataset(index):
    fileClassification = pd.read_csv(index, sep=' ', names=['is_spam', 'email_filename'])

    fileClassification['is_spam'] = fileClassification['is_spam'].map({'spam': 1, 'ham': 0})
    fileClassification['email_filename'] = [x.replace('../', '') for x in fileClassification['email_filename']]
    fileClassification['text'] = ''

    dataFrameResult = parallelProcessing(fileClassification)
    #convert the result into a csv file
    dataFrameResult.to_csv(os.path.join(path, csvFileName))


def parallelProcessing(fileClassification): 
    #split the array into 2 parts
    fileSubArray = np.array_split(fileClassification,5)
    
    # Create new threads
    #thread1 = myThread(1, "email-Thread-1", fileClassification)
    thread1 = myThread(1, "email-Thread-1", fileSubArray[0])
    thread2 = myThread(2, "email-Thread-2", fileSubArray[1])
    thread3 = myThread(3, "email-Thread-3", fileSubArray[2])
    thread4 = myThread(4, "email-Thread-4", fileSubArray[3])
    thread5 = myThread(5, "email-Thread-5", fileSubArray[4])

    # Start new Threads
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()

    # Join the threads
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()


    #merge all the data that was collected
    resultingFrames = [thread1.files,thread2.files,thread3.files,thread4.files,thread5.files]
    #resultingFrames = [thread1.files]
    #concatinate all of the resulting frames
    dataFrameResult = pd.concat(resultingFrames)
    #dataFrameResult.replace('', np.nan, inplace=True)
    #dataFrameResult.dropna(how='any', subset=['text'], inplace=True)
    #print(dataFrameResult)
    return dataFrameResult

#Threading class in order to create a parallel processing
#source: https://www.tutorialspoint.com/python/python_multithreading.htm
class myThread (threading.Thread):
   def __init__(self, threadID, name, files):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.files = files

   def run(self):
        print ("Starting " + self.name)
        readEmail(self.name, self.files)
        print ("Exiting " + self.name)


if __name__ == "__main__":
   #readEmail(os.path.join(path, 'full/index'))
   #read the email one by one
   preprocessDataset(os.path.join(path, indexFilePath))
   
#    spamClassifier = pd.read_csv(os.path.join(path, csvFileName), usecols=['is_spam', 'text'])
#    spamClassifier['text'] = spamClassifier['text'].astype(str)

#    spamClassifier['text'] = spamClassifier['text'].apply(deleteNonASCII)
#    spamClassifier['text'] = spamClassifier['text'].apply(cleanhtml)

#    spamClassifier['text'].replace('', np.nan, inplace=True)
#    spamClassifier['text'].dropna(how='any', inplace=True)
#    #print(spamClassifier.head(10))

#    print(spamClassifier.groupby('text').describe())

