import os
import email
import mailparser


def readEmail():
    path = r'C:\Users\Aldrin\Desktop\school_folders\CS191-ML\trec07p\data'
    folder = os.fsencode(path)

    for file in sorted(os.listdir(folder)):
        filename = os.fsdecode(file)

        
        #mail = mailparser.parse_from_file(path+"\\"+filename)
        #print(mail.body)


readEmail()

