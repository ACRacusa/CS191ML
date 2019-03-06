library(dplyr)
library(tm)
library(knitr)
library(RWeka)
library(caret)

preprocessed <-
  read.csv(file = "C:/Users/Aldrin/Desktop/school_folders/CS191-ML/trec07p/preprocessed.csv", head =
             TRUE, sep = ",")
preprocessedDF <- as.data.frame(preprocessed)
preprocessedReduced <-
  read.csv(file = "C:/Users/Aldrin/Desktop/school_folders/CS191-ML/trec07p/preprocessed-red.csv", head =
             TRUE, sep = ",")
preprocessedDFReduced <- as.data.frame(preprocessedReduced)
#############--Training the General Vocabulary--######################
set.seed(101) # Set Seed so that same sample can be reproduced in future also
# Now Selecting 75% of data as sample from total 'n' rows of the data  
sample <- sample.int(n = nrow(preprocessedDF), size = floor(.70*nrow(preprocessedDF)), replace = F)
reduceSample <- sample.int(n = nrow(preprocessedDFReduced), size = floor(.70*nrow(preprocessedDFReduced)), replace = F)
GeneralVocabTrain <- preprocessedDF[sample, ]
GeneralVocabtest  <- preprocessedDF[-sample, ]
ReducedVocabTrain <- preprocessedDFReduced[reduceSample, ]
ReducedVocabTest <-preprocessedDFReduced[-reduceSample, ]

#General vocab Training/test set
GenTargetTrain <- GeneralVocabTrain$tokenDF.is_spam
GenTargetTest <- GeneralVocabtest$tokenDF.is_spam
GeneralVocabTrain$tokenDF.is_spam <- NULL
GeneralVocabtest$tokenDF.is_spam <- NULL

#Reduced Training/test set
RedTargetTrain <-ReducedVocabTrain$tokenDF.is_spam
RedTargetTest <- ReducedVocabTest$tokenDF.is_spam
ReducedVocabTrain$tokenDF.is_spam <- NULL
ReducedVocabTest$tokenDF.is_spam <- NULL


naiveBayesFunction <- function(l,vocabTrain,vocabTest,targetTrain,targetTest,genOrRed){
  
  naiveLambda <- l;
  laplaceSmoothing <- naiveLambda * 2
  
  ######without laplace smoothing
  #get the number of spam and ham in the target training set
  spamCount <- sum(targetTrain == 1,na.rm=TRUE)
  hamCount <- sum(targetTrain == 0,na.rm=TRUE)
  totalCount <- spamCount + hamCount
  
  #get the features and its length
  spamHamFeatures <- colnames(vocabTrain)
  spamHamFeaturesLength <- length(spamHamFeatures)
  
  #get the proby of ham and spam over the total count
  probyHam <- (hamCount + naiveLambda) / (totalCount + laplaceSmoothing)
  probySpam <- (spamCount + naiveLambda) / (totalCount + laplaceSmoothing)
  
  hamLocTargetTrainIndex <- which(targetTrain == 0)
  spamLocTargetTrainIndex <- which(targetTrain == 1)
  #Get the values of the vocab that contains the ham and the spam
  X_Ham <- vocabTrain[hamLocTargetTrainIndex,]
  X_Spam <-vocabTrain[spamLocTargetTrainIndex,]
  
  newLaplaceSmoothing <- naiveLambda * spamHamFeaturesLength
  
  proby_feature_is_1_given_Ham <- c()
  proby_feature_is_1_given_Spam <- c()
  
  for (f in spamHamFeatures) {
    # print(nrow(X_Ham[which(X_Ham[as.character(f)] == 1),]))
    # break
    xHamFeaturesIndex <- which(X_Ham[as.character(f)] == 1)
    xHamFeaturesLoc <- X_Ham[xHamFeaturesIndex,]
    numberOfHamWithF = nrow(xHamFeaturesLoc)
    proby_f_is_1_given_Ham <- (numberOfHamWithF + naiveLambda) / (hamCount + newLaplaceSmoothing)
    proby_feature_is_1_given_Ham <- append(proby_feature_is_1_given_Ham,proby_f_is_1_given_Ham)
    
    xSpamFeaturesIndex <- which(X_Spam[as.character(f)] == 1)
    xSpamFeaturesLoc <- X_Spam[xSpamFeaturesIndex,]
    numberOfSpamWithF = nrow(xSpamFeaturesLoc)
    proby_f_is_1_given_Spam <- (numberOfSpamWithF + naiveLambda) / (spamCount + newLaplaceSmoothing)
    proby_feature_is_1_given_Spam <- append(proby_feature_is_1_given_Spam,proby_f_is_1_given_Spam)
  }
  
  prediction <- c()
  for (i in 1:length(vocabTest[,1])) {
    #given = list(X_test.iloc[i].values)
    given <- vocabTest[i,]
    
    proby_features_given_ham = 1
    proby_features_given_spam = 1
    
    for (j in 1:spamHamFeaturesLength) {
      if (given[j] == 1) {
        
        proby_features_given_ham = proby_features_given_ham * proby_feature_is_1_given_Ham[j]
        proby_features_given_spam = proby_features_given_spam * proby_feature_is_1_given_Spam[j]
      } else{
        proby_features_given_ham = proby_features_given_ham * (1 - proby_feature_is_1_given_Ham[j])
        proby_features_given_spam = proby_features_given_spam * (1 - proby_feature_is_1_given_Spam[j])
      }
    }
    
    proby_ham_given_features = proby_features_given_ham * probyHam
    proby_spam_given_features = proby_features_given_spam * probySpam
    
    if(proby_ham_given_features > proby_spam_given_features){
      prediction <- append(prediction,0)
    }else{
      prediction <- append(prediction,1)
    }
    
  }#end genVocabTest loop
  accuracy = 0
  for(i in 1:length(targetTest)){
    if(targetTest[i] == prediction[i]){
      accuracy = accuracy + 1
    }
  }
  
  if(l!=0){
    cat(genOrRed," Vocabulary with Laplace Smoothing\n")
  }else{
    cat(genOrRed," Vocabulary without Laplace Smoothing\n")
  }
  
  total_accuracy = accuracy/length(prediction)
  cat("Accuracy: ",total_accuracy,"\n")
  
  xtab <- table(prediction,targetTest)
  xtab
  print("Confusion Matrix")
  print(confusionMatrix(xtab))
  
  print("Precision")
  print(precision(xtab))
  print("Recall")
  print(recall(xtab))
}

#General Vocab
naiveBayesFunction(1,GeneralVocabTrain,GeneralVocabtest,GenTargetTrain,GenTargetTest,"General")
naiveBayesFunction(0,GeneralVocabTrain,GeneralVocabtest,GenTargetTrain,GenTargetTest,"General")
#Reduced Vocab
naiveBayesFunction(1,ReducedVocabTrain,ReducedVocabTest,RedTargetTrain,RedTargetTest,"Reduced")
naiveBayesFunction(0,ReducedVocabTrain,ReducedVocabTest,RedTargetTrain,RedTargetTest,"Reduced")
