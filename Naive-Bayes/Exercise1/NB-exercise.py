import os
import email
import mailparser
import pandas as pd
import numpy as np

def readEmail(fileClass):
    path = r'C:\Users\Aldrin\Desktop\school_folders\CS191-ML\trec07p\data'
    folder = os.fsencode(path)

    for file in sorted(os.listdir(folder)):
        filename = os.fsdecode(file)

        
        #mail = mailparser.parse_from_file(path+"\\"+filename)
        #print(mail.body)

#readEmail()

def readClassification():
        path = r'C:\Users\Aldrin\Desktop\school_folders\CS191-ML\trec07p\full\index'
 
        #folder = os.fsencode(path)
        file = pd.read_csv(path, sep=' ',header=None)
        
        print(file.loc[:,0])
        #firstCol = np.asarray(file[1:])
        #print (firstCol)
        readEmail(file)

if __name__ == "__main__":
        readClassification()