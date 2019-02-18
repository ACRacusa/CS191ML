import email, re, os, threading
import pandas as pd
import numpy as np
import mailparser

path = 'C:/Users/Aldrin/Desktop/school_folders/CS191-ML/trec07p/'
indexFilePath = 'full/index'
csvFileName = 'parsedEmails.csv'
exitFlag = 0

# def readEmail(thread_name, email_frame):  
#     folder = os.fsencode(path)

#     for file in sorted(os.listdir(folder)):
#         filename = os.fsdecode(file)
#         #print(fileClass.loc[i,0])
#         print(extract_email_from_file(path+"/"+filename))

#iterate through the data frame rows 
#Source: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
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

#Preprocess the "/full/index" to convert it to csv file
def preprocessDataset(index):
    fileClassification = pd.read_csv(index, sep=' ', names=['is_spam', 'email_filename'])
    fileClassification['is_spam'] = fileClassification['is_spam'].map({'spam': 1, 'ham': 0})
    fileClassification['email_filename'] = [x.replace('../', '') for x in fileClassification['email_filename']]
    fileClassification['text'] = ''

    parallelProcessing(fileClassification)
    #print(fileClassification)

def parallelProcessing(fileClassification): 
    #split the array into 2 parts
    fileSubArray = np.array_split(fileClassification,2)
    
    # Create new threads
    thread1 = myThread(1, "email-Thread-1", fileSubArray[0])
    thread2 = myThread(2, "email-Thread-2", fileSubArray[1])

    # Start new Threads
    thread1.start()
    thread2.start()

    # Join the threads
    thread1.join()
    thread2.join()

    #merge all the data that was collected
    resultingFrames = [thread1.files,thread2.files]
    #concatinate all of the resulting frames
    dataFrameResult = pd.concat(resultingFrames)
    
    #convert the result into a csv file
    dataFrameResult.to_csv(os.path.join(path, csvFileName))
    #print(dataFrameResult)

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
 

#clean the html tags from the string
#Source: https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

#Extract the emailfile from the given filename and then cleans the html tags
def parseEmailFromFile(emailFile):
    mail = mailparser.parse_from_file(emailFile)
    email_body = cleanhtml(mail.body)
    finalCleanEmail = email_body.replace('\n',' ').strip()
    return finalCleanEmail

if __name__ == "__main__":
   #readEmail(os.path.join(path, 'full/index'))
   preprocessDataset(os.path.join(path, indexFilePath))