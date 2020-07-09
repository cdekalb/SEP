library(RWeka)
library(tidyverse)
library(caret)
library(rpart.plot)
library(ggplot2)

file1 <- read.arff("1year.arff")
file2 <- read.arff("2year.arff")
file3 <- read.arff("3year.arff")
file4 <- read.arff("4year.arff")
file5 <- read.arff("5year.arff")

set.seed(5900)

# The project thus far has been only on the first year of data. The reasoning is that the attributes in the data may take
# on different meanings from year to year. If it can be assumed that the attributes are consistent, then the models can
# be scaled up to use each year of data together as one dataset.

file1 <- as.data.frame(file1)

# Check missingness
colSums(is.na(file1))

# 23% missing for Attr21 and 39% missing for Attr37
# Too many missing from those attributes to impute, continue with them removed
newFile1 <- cbind(file1[1:20], file1[22:36], file1[38:65])
newFile1 <- as.data.frame(newFile1)
colSums(is.na(newFile1))

# The next attribute with the most missingness is Attr27 with only 4% missing
# Impute missing values for each column with the median value of each column
for (i in 1:ncol(newFile1)) {
  if (colSums(is.na(newFile1))[i] != 0) {
    colMedian <- median(na.omit(newFile1[,i]))
    newFile1[,i][is.na(newFile1[,i])] <- colMedian
  }
}

# Some basic plots to get an idea of any relationships between the first several attributes
plot1 <- ggplot(newFile1, aes(x = Attr1, y = Attr2, color = class)) + theme_bw() + geom_point()
plot1
plot1Zoom <- ggplot(newFile1, aes(x = Attr1, y = Attr2, color = class)) + theme_bw() + geom_point() + xlim(-5, 5) + ylim(-5,5)
plot1Zoom
plot2 <- ggplot(newFile1, aes(x = Attr3, y = Attr4, color = class)) + theme_bw() + geom_point()
plot2
plot2Zoom <- ggplot(newFile1, aes(x = Attr3, y = Attr4, color = class)) + theme_bw() + geom_point() + xlim(-25, 5) + ylim(-5,100)
plot2Zoom

# From these plots, there does not seem to be an obvious relationship between Attr1 and Attr2 or Attr3 and Attr4.

# Get principal components for the first year of data
pcaAttributes <- prcomp(newFile1[,1:62], scale. = TRUE)
summary(pcaAttributes)

# From the summary of the principal components for the data, we see that the first 15 principal components explain 90% of
# the variance in the data, thus we could reduce the dimensionality of the predictor variables from 62 to 15 with only a
# 10% loss in variance explanation.

# Assign IDs to each row to split for training and testing datasets
IDs <- 1:nrow(newFile1)
newFile1 <- cbind(newFile1, IDs)

# Create a randomly selected training dataset
numTrainingIDs <- floor(7027*.7)
trainFile1 <- newFile1[sample(nrow(newFile1), numTrainingIDs),]
trainFile1$class <- ifelse(trainFile1$class==1, "Yes", "No")
head(trainFile1)

# Create the corresponding test dataset
testFile1IDs <- newFile1[,64][is.na(pmatch(newFile1[,64],trainFile1[,64]))]
testFile1 <- newFile1[newFile1$IDs %in% testFile1IDs,]
testFile1$class <- ifelse(testFile1$class==1, "Yes", "No")
testFile1WithClass <- testFile1
testFile1$class <- NA
head(testFile1)

######## Decision tree model ########

# Set the training control parameters to be used for each model
control = trainControl(method = "repeatedcv" 
                       ,number = 3
                       ,classProbs = TRUE 
                       ,verboseIter = TRUE)

decisionTree = train(class ~ .
                     ,data = trainFile1 %>% select(-c(IDs)) 
                     ,method = "ctree"
                     ,trControl = control
                     ,tuneLength = 10) 
decisionTree

# Print the confusion matrix and its corresponding statistics
decisionTreePrediction <- predict(decisionTree, newdata = testFile1WithClass)
testFile1WithClass$class <- as.factor(testFile1WithClass$class)
confusionMatrix(decisionTreePrediction, testFile1WithClass$class)

# With a specificity value of 0, the model did not correctly predict a single instance of class = 1.

######## Recursive Partitioning and Regression Trees decision tree model ########

# Train the decision tree
decisionTreeRPart = train(class ~ .
                          ,data = trainFile1 %>% select(-c(IDs)) 
                          ,method = "rpart"
                          ,trControl = control
                          ,tuneLength = 10) 
decisionTreeRPart

# Print the confusion matrix and its corresponding statistics
decisionTreeRPartPrediction <- predict(decisionTreeRPart, newdata = testFile1WithClass)
confusionMatrix(decisionTreeRPartPrediction, testFile1WithClass$class)

# Print the decision tree
prp(decisionTreeRPart$finalModel, box.palette = "Greens", tweak = 1.5)

# The accuracy value of 0.972 seems promising at face value, but the specificity value of 0.4409 implies that the model is
# correct less than half of the time when predicting the value of class when class = 1.

# Generalized linear model
# glm = train(class ~ .
#             ,data = trainFile1 %>% select(-c(IDs)) 
#             ,method = "glm" 
#             ,trControl = control
#             ,preProcess = c("center", "scale")) 

# Attempting to create a generalized linear model results in error saying "fitted probabilities numerically 0 or 1 
# occurred."

# Random forest model
randomForest = train(class ~ .
                ,data = trainFile1 %>% select(-c(IDs))
                ,method = "rf"
                ,trControl = control)
randomForest

# Print the confusion matrix and its corresponding statistics
randomForestPrediction <- predict(randomForest, newdata = testFile1WithClass)
confusionMatrix(randomForestPrediction, testFile1WithClass$class)

# The random forest model's accuracy value of 0.973 is approximately equal to that of the decision tree model, but its
# specificity value of 0.4301 is less.

# Before using a set seed to choose the training dataset, the decision tree and random forest models were run and the
# random forest model had a higher accuracy and specificity value. This could be due to to the comparitively small 
# training dataset of 4918 entries. A more stable model might arise if the yearly data were concatenated.

# Support Vector Machine model
supportVectorMachine = train(class ~ .
                 ,data = trainFile1 %>% select(-c(IDs))
                 ,method = "svmRadial"
                 ,trControl = control)
supportVectorMachine

supportVectorMachinePrediction <- predict(supportVectorMachine, newdata = testFile1WithClass)
confusionMatrix(supportVectorMachinePrediction, testFile1WithClass$class)

# With a specificity value of 0.0215, the model very rarely correctly predicts when class = 1.

# K-nearest neighbor model
KNN = train(class ~ .
            ,data = trainFile1 %>% select(-c(IDs))
            ,method = "kknn"
            ,trControl = control)
KNN

KNNPrediction <- predict(KNN, newdata = testFile1WithClass)
confusionMatrix(KNNPrediction, testFile1WithClass$class)

# With a specificity value of 0.0108, the model very rarely correctly predicts when class = 1.
