# R script of cross-validated classification (ridge logistic regression) for predicting future abeta positivity 
# Author: Elaheh Moradi, University of Easatern Finland, Kuopio ,Finland (elaheh.moradi@uef.fi)
#last updated 22.06.2023
#
# Requirements:
# R version 4.1.1 or later
#install.packages("glmnet")
#install.packages("caret", dependencies = c("Depends", "Suggests"))

#
# Usage: 
# Set working directory to directory containing R script
# source('classification_RLR.R')

#Parameters: 
#Xdata: input matrix, of dimension nobs x nvars; each row is an observation vector
#label: response variable with two levels
#aux_data: auxiliary data for using only in training phase
#aux_label: label information for  auxiliary data 
#seed: seed number for reproducing the results

#Returned values:
#A list of 3 items (pred, prob, and coeffs)
#pred: predicted label
#prob: predicted probability
#coeffs: a matrix of coefficient values derived from K fold experiments
###########################
classification_RLR= function(Xdata, label, aux_data, aux_label, seed){
  library(glmnet)
  library(caret)
  set.seed(seed)
  folds<- createFolds(label, k = 10, list = TRUE, returnTrain = FALSE)
  allAct<- vector()
  allPred= vector()
  allProb= vector()
  RID_pred= vector()
  coeffs= matrix(0, nrow = ncol(Xdata), ncol= 10)
  RID= rownames(Xdata)
  for (i in 1:length(folds)){
    print(i)
    ind<- folds[[i]]
    Xtrain= Xdata[-ind,]
    Ytrain=label[-ind]
    Xtest= Xdata[ind,]
    Ytest= label[ind]
    RID_pred= c(RID_pred,RID[ind])
    #   # 
    Xtrain<- rbind(Xtrain, aux_data)
    Ytrain<- factor(c(Ytrain,aux_label))
    
    normParam <- preProcess(Xtrain)
    Xtrain  <- predict(normParam, Xtrain)
    Xtest<- predict(normParam, Xtest)
    
    model=cv.glmnet(Xtrain, Ytrain, family="binomial",type.measure="class", alpha= 0)
    tmp <- as.vector(coef(model, s = "lambda.min"))[-1]
    coeffs[,i]=tmp
    
    ypred<- predict(model, Xtest,  s="lambda.min", type= "class")
    allAct= c(allAct, Ytest)
    allPred= c(allPred,ypred)
    
    prob=predict(model, Xtest, s="lambda.min",type= "response")
    allProb= c(allProb,prob)
    
  }
  rownames(coeffs)= colnames(Xdata)
  ind= match(rownames(Xdata), RID_pred)
  pred= allPred[ind]
  prob= allProb[ind]
  results= list(pred=pred, prob=prob, coeffs=coeffs)
  return(results)
  
}