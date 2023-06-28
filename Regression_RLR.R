#R script of cross-validated regression (ridge linear regression) for predicting future abeta value(CSFAB42, PET global SUVR) 
#Author: Elaheh Moradi, University of Easatern Finland, Kuopio ,Finland (elaheh.moradi@uef.fi)
#last updated 22.06.2023
#
#Requirements:
#R version 4.1.1 or later
#install.packages("glmnet")
#install.packages("caret", dependencies = c("Depends", "Suggests"))

#
# Usage: 
# Set working directory to directory containing R script
# source('classification_RLR.R')

#Parameters: 
#Xdata: input matrix, of dimension nobs x nvars; each row is an observation vector
#label: response variable 
#seed: seed number for reproducing the results

#Returned values:
#A list of 2 items (pred and coeffs)
#pred: predicted label
#coeffs: a matrix of coefficient values derived from K fold experiments
###########################
Regression_RLR= function(Xdata, label,seed){
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
   
    normParam <- preProcess(Xtrain)
    Xtrain  <- predict(normParam, Xtrain)
    Xtest<- predict(normParam, Xtest)
    
    model=cv.glmnet(Xtrain, Ytrain,alpha= 0)
    tmp <- as.vector(coef(model, s = "lambda.min"))[-1]
    coeffs[,i]=tmp
    ypred<- predict(model, Xtest,  s="lambda.min")
    allAct= c(allAct, Ytest)
    allPred= c(allPred,ypred)
    
  }
  rownames(coeffs)= colnames(Xdata)
  ind= match(rownames(Xdata), RID_pred)
  pred= allPred[ind]

  results= list(pred=pred, coeffs=coeffs)
  return(results)
  
}