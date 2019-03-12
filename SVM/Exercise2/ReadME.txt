



There is one(1) SVM Classifier presented here: 
	- classifying emails as either ham or spam  (in Exercise1 folder)
	
      1. SVM Classifier in classifying emails as either ham or spam  (in Exercise1 folder)
	- There are 3 python scripts here:
		*parse-email.py
			- This python script parses the given emails in the dataset in a concurrent manner, and then saves the output CSV file with a filename: "parsedEmails.csv".
		*svm-bow.py
			-This python script does the initial preprocessing of emails. It involves removing non-ASCII, HTML tags, strings with numeric characters embedded, remove punctuations, lowercase the document, perform initial lemmatization, and remove stopwords. And then, saves the array-like output to a CSV file with a filename: "tokenizedData.csv"

		*svm-model.py
			-The "tokenizedData.csv" will be converted into a TF-IDF format and then, using a pipeline, the SVM algorithm will be performed with the training and test set.
	

Output files are described in Spam_Classification_using_Support_Vector_Machine.pdf
