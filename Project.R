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
colSums(is.na(newFile1))


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

######## Feature Importance Exploration ########

# Get principal components for the first year of data
pcaAttributes <- prcomp(newFile1[,1:62], scale. = TRUE)
summary(pcaAttributes)

# From the summary of the principal components for the data, we see that the first 15 principal components explain 90% of
# the variance in the data, thus we could reduce the dimensionality of the predictor variables from 62 to 15 with only a
# 10% loss in variance explanation.

control <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
model <- train(class ~.
               ,data = newFile1
               ,method = "lvq"
               ,trControl = control)
varImportance <- varImp(model, scale = FALSE)
print(varImportance)
plot(varImportance)

# Create a randomly selected training dataset
trainData1Samples <- newFile1$class %>% createDataPartition(p = 0.7, list = FALSE)
trainData1 <- newFile1[trainData1Samples,]

# Create the corresponding test dataset
testData1 <- newFile1[-trainData1Samples,]

length(trainData1$class[trainData1$class == 1])
length(trainData1$class[trainData1$class == 0])

# Use oversampling to account for the high imbalance
overBalancedTrainData1 <- sample_n(trainData1[trainData1$class == 1,], 
                                   size = nrow(trainData1[trainData1$class == 0,]), 
                                   replace = TRUE)
balancedTrainData1 <- rbind(overBalancedTrainData1, trainData1[trainData1$class == 0,])
balancedTrainData1$class <- as.numeric(balancedTrainData1$class)
testData1$class <- as.numeric(testData1$class)
balancedTrainData1 <- as.data.frame(balancedTrainData1)
testData1 <- as.data.frame(testData1)

# balancedTrainData1$class[balancedTrainData1$class == 1] <- 0
# balancedTrainData1$class[balancedTrainData1$class == 2] <- 1
# testData1$class[testData1$class == 1] <- 0
# testData1$class[testData1$class == 2] <- 1

# length(balancedTrainData1$class[balancedTrainData1$class == 1])
# length(balancedTrainData1$class[balancedTrainData1$class == 0])

######## Preliminary Linear Model For Determining Multicollinearity ########

linearModel <- lm(class ~ ., data = balancedTrainData1)
linearModelPrediction <- linearModel %>% predict(testData1)

alias(lm(class ~ ., data = balancedTrainData1))

linearModel <- lm(class ~ . -Attr14 -Attr18, data = balancedTrainData1)
linearModelPrediction <- linearModel %>% predict(testData1)

data.frame(
  RMSE = RMSE(linearModelPrediction, testData1$class),
  R2 = R2(linearModelPrediction, testData1$class)
)

car::vif(linearModel)

linearModel <- lm(class ~ Attr5 + Attr15 + Attr27 + Attr28 + Attr29 + Attr41 + Attr55 + Attr57 + Attr59, 
                  data = balancedTrainData1)
linearModelPrediction <- linearModel %>% predict(testData1)

data.frame(
  RMSE = RMSE(linearModelPrediction, testData1$class),
  R2 = R2(linearModelPrediction, testData1$class)
)

balancedTrainData1$class[balancedTrainData1$class == 1] <- 0
balancedTrainData1$class[balancedTrainData1$class == 2] <- 1
testData1$class[testData1$class == 1] <- 0
testData1$class[testData1$class == 2] <- 1

balancedTrainData1$class <- as.factor(balancedTrainData1$class)
testData1$class <- as.factor(testData1$class)

# Set the training control parameters to be used for each model
control = trainControl(method = "repeatedcv" 
                       ,number = 5
                       ,repeats = 2)

######## Linear Discriminant Analysis ########

# Train the model
fit.lda = train(class ~ Attr5 + Attr15 + Attr27 + Attr28 + Attr29 + Attr41 + Attr55 + Attr57 + Attr59 
                  ,data = balancedTrainData1
                  ,method = "lda"
                  ,metric = "Accuracy"
                  ,trControl = control)

# Print the confusion matrix and its corresponding statistics
fit.ldaPrediction <- predict(fit.lda, newdata = testData1)
confusionMatrix(fit.ldaPrediction, testData1$class)
F_meas(fit.ldaPrediction, reference = testData1$class)

######## Generalized Linear Model ########

# Train the model
fit.glm = train(class ~ Attr5 + Attr15 + Attr27 + Attr28 + Attr29 + Attr41 + Attr55 + Attr57 + Attr59 
                ,data = balancedTrainData1
                ,method = "glm"
                ,metric = "Accuracy"
                ,trControl = control)

# Print the confusion matrix and its corresponding statistics
fit.glmPrediction <- predict(fit.glm, newdata = testData1)
confusionMatrix(fit.glmPrediction, testData1$class)
F_meas(fit.glmPrediction, reference = testData1$class)

# Training the generalized linear model results in errors saying "fitted probabilities numerically 0 or 1 
# occurred," but a working model is still produced.

######## Recursive Partitioning and Regression Trees ########

# Train the model
fit.rpart = train(class ~ Attr5 + Attr15 + Attr27 + Attr28 + Attr29 + Attr41 + Attr55 + Attr57 + Attr59 
                  ,data = balancedTrainData1
                  ,method = "rpart"
                  ,metric = "Accuracy"
                  ,trControl = control)

# Print the confusion matrix and its corresponding statistics
fit.rpartPrediction <- predict(fit.rpart, newdata = testData1)
confusionMatrix(fit.rpartPrediction, testData1$class)
F_meas(fit.rpartPrediction, reference = testData1$class)

# Print the decision tree
prp(fit.rpart$finalModel, box.palette = "Greens", tweak = 1.5)

######## Random Forest ########

# Train the model
fit.rf = train(class ~ Attr5 + Attr15 + Attr27 + Attr28 + Attr29 + Attr41 + Attr55 + Attr57 + Attr59
                     ,data = balancedTrainData1
                     ,method = "rf"
                     ,metric = "Accuracy"
                     ,trControl = control)


# Print the confusion matrix and its corresponding statistics
fit.rfPrediction <- predict(fit.rf, newdata = testData1)
confusionMatrix(fit.rfPrediction, testData1$class)
F_meas(fit.rfPrediction, reference = testData1$class)

######## Support Vector Machine ########

# Train the model
fit.svm = train(class ~ Attr5 + Attr15 + Attr27 + Attr28 + Attr29 + Attr41 + Attr55 + Attr57 + Attr59
                ,data = balancedTrainData1
                ,method = "svmRadial"
                ,metric = "Accuracy"
                ,trControl = control)

# Print the confusion matrix and its corresponding statistics
fit.svmPrediction <- predict(fit.svm, newdata = testData1)
confusionMatrix(fit.svmPrediction, testData1$class)
F_meas(fit.svmPrediction, reference = testData1$class)

results <- resamples(list(lda=fit.lda, glm=fit.glm, svm=fit.svm, rpart=fit.rpart, rf=fit.rf))
summary(results)

