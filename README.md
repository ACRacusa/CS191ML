# CS191ML
**Exercise 1: Naive Bayes**




There is one(1) Bayesian Classifier presented here: 
	- classifying emails as either ham or spam  (in Exercise1 folder)
	
      1. Bayesian Classifier in classifying emails as either ham or spam  (in Exercise1 folder)
	- There are 3 python scripts and 2 R scripts here:
		*parse-email.py
			- This python script parses the given emails in the dataset in a concurrent manner, and then saves the output CSV file with a filename: "parsedEmails.csv".
		*NB-exercise.py
			-This python script does the initial preprocessing of emails. It involves removing non-ASCII, HTML tags, strings with numeric characters embedded, remove punctuations, lowercase the document, perform initial lemmatization, and remove stopwords. And then, saves the array-like output to a CSV file with a filename: "tokenizedData.csv"
		*tokenize.R
			-This R script will then parse the "tokenizedData.csv" and then performs another preprocessing to double check some leftover unremoved unnecessary characters. This involves remove unnecessary white spaces, removing non-ASCII characters, and perform Stemming on each words. After that, it converts the dataframe into a Bag-of-words type of dataframe. This means that given a number of features, every document will check if a word is the same as the given feature, if so, then it is labeled "1" else, "0". And repeat this until every words in the whole dataset has been converted, and saves the Output to a CSV file with a filename: "preprocessed.csv" for General Vocabulary (which contains 3371 features) and "preprocessed-red.csv" for Reduced Vocabulary (which contains 237 features).

		*ManualModel.py and train.R
			-There is a Python and R version of the training and classification of the spam/ham. These two scripts perform the Naive Bayes Algorithm in the "preprocessed.csv/preprocessed-red.csv" dataset.
	

Output files are described in Spam_Classification_of_emails_using_Naive_Bayes_Algorithm.pdf

**Exercise 2: SVM**
There is one(1) SVM Classifier presented here: 
	- classifying emails as either ham or spam  (in Exercise2 folder)
	
      1. SVM Classifier in classifying emails as either ham or spam  (in Exercise2 folder)
	- There are 3 python scripts here:
		*parse-email.py
			- This python script parses the given emails in the dataset in a concurrent manner, and then saves the output CSV file with a filename: "parsedEmails.csv".
		*svm-bow.py
			-This python script does the initial preprocessing of emails. It involves removing non-ASCII, HTML tags, strings with numeric characters embedded, remove punctuations, lowercase the document, perform initial lemmatization, and remove stopwords. And then, saves the array-like output to a CSV file with a filename: "tokenizedData.csv"

		*svm-model.py
			-The "tokenizedData.csv" will be converted into a TF-IDF format and then, using a pipeline, the SVM algorithm will be performed with the training and test set.
	

Output files are described in Spam_Classification_using_Support_Vector_Machine.pdf

**Exercise 3: ICA**




There is one(1) ICA presented here: 
	- Performing Blind source separation in two audio files (in codes folder)
	
      1. ICA for the two audio files
	- There are 3 python scripts here:
		*model.py
			- This python script performs the Independent Component Analysis by using the FastICA. First it combines the two separate audio files by using the zip() function which returns an iterative tuple of the mixed file, and then performs FastICA to separate the independent components which are then saved as a WAV file.
		*plot-model.py
			-This python script shows the visualization of the two audio source files

		*plot-results.py
			-This python script shows the visualization of the two Independent Components that are produced from the model.py
	

Output files are described in Blind_Source_Separation_using_Independent_Component_Analysis__ICA_.pdf

