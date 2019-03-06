library(dplyr)
library(tm)
library(knitr)
library(RWeka)
##################
 #https://www.kaggle.com/rhitamjeet/bag-of-words-nlp-tm-rweka-logistic-regression
#https://www.kaggle.com/amhchiu/bag-of-ingredients-in-r
# install.packages("tm")  # for text mining
# install.packages("SnowballC") # for text stemming
# install.packages("wordcloud") # word-cloud generator 
# install.packages("RColorBrewer") # color palettes
# Load
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
#########
tokenizedData <-
  read.csv(file = "C:/Users/Aldrin/Desktop/school_folders/CS191-ML/trec07p/tokenizedData.csv", head =
             TRUE, sep = ",")
tokenDF <- as.data.frame(tokenizedData)
set.seed(101) # Set Seed so that same sample can be reproduced in future also
# Now Selecting 75% of data as sample from total 'n' rows of the data  
sample <- sample.int(n = nrow(tokenDF), size = floor(.75*nrow(tokenDF)), replace = F)
word_train <- tokenDF[sample, ]
word_test  <- tokenDF[-sample, ]

full = bind_rows(word_train,word_test)
dim(word_train)
dim(word_test)
str(full)

full$text = gsub(full$text, pattern = '<br />', replacement = ' ')
text = VCorpus(VectorSource(full$text))        # Creating a Corpus of reviews
text = tm_map(text,content_transformer(tolower))  # Converting to lower case
text = tm_map(text,removeNumbers)                # Removing numbers
text = tm_map(text,removePunctuation)            # Removing Punctuations
text = tm_map(text,removeWords,stopwords())      #to remove common words
text = tm_map(text,stemDocument)                 #to convert words back to root words.
text = tm_map(text,stripWhitespace)              #to remove white spaces
head(text,5)

dtm1 = DocumentTermMatrix(text)
dtm1 = removeSparseTerms(dtm1,0.997)
#dtm1 = removeSparseTerms(dtm1,0.90)
dataset1 = as.data.frame(as.matrix(dtm1))
dataset_counts1 = as.data.frame(colSums(dataset1))
dataset_counts1$word = rownames(dataset_counts1)

colnames(dataset_counts1) = c("count","word")
dataset_counts1 = dataset_counts1[c(2,1)] 
dataset_counts1 = dataset_counts1 %>% arrange(-count)

# write.table(dataset1, file = "dtm.csv", na="", sep=",")
#############
# m <- as.matrix(dtm1)
# v <- sort(rowSums(m),decreasing=TRUE)
# d <- data.frame(word = names(v),freq=v)
# #create word cloud
# set.seed(1234)
# wordcloud(words = dataset_counts1$word, freq = dataset_counts1$count, min.freq = 1,
#           max.words=200, random.order=FALSE, rot.per=0.35, 
#           colors=brewer.pal(8, "Dark2"))
preprocessedData <- cbind(tokenDF$is_spam,dataset1)
write.table(preprocessedData, file = "C:/Users/Aldrin/Desktop/school_folders/CS191-ML/trec07p/preprocessed.csv",row.names = FALSE, na="", sep=",")


##########for reduced################
dtm1 = DocumentTermMatrix(text)
dtm1 = removeSparseTerms(dtm1,0.95)
#dtm1 = removeSparseTerms(dtm1,0.90)
dataset1 = as.data.frame(as.matrix(dtm1))
dataset_counts1 = as.data.frame(colSums(dataset1))
dataset_counts1$word = rownames(dataset_counts1)

colnames(dataset_counts1) = c("count","word")
dataset_counts1 = dataset_counts1[c(2,1)] 
dataset_counts1 = dataset_counts1 %>% arrange(-count)

#write.table(dataset1, file = "dtm.csv", na="", sep=",")
#############
# m <- as.matrix(dtm1)
# v <- sort(rowSums(m),decreasing=TRUE)
# d <- data.frame(word = names(v),freq=v)
# #create word cloud
# set.seed(1234)
# wordcloud(words = dataset_counts1$word, freq = dataset_counts1$count, min.freq = 1,
#           max.words=200, random.order=FALSE, rot.per=0.35, 
#           colors=brewer.pal(8, "Dark2"))
preprocessedData <- cbind(tokenDF$is_spam,dataset1)
write.table(preprocessedData, file = "C:/Users/Aldrin/Desktop/school_folders/CS191-ML/trec07p/preprocessed-red.csv",row.names = FALSE, na="", sep=",")