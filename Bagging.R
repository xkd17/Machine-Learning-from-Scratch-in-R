# Multiclassification by Bagging Decision Tree Models.
# Author: Michael Ruochen Zeng, micalzrc@gmail.com
library(rpart)
library(pryr)
library(mlbench)
library(data.table)
rm(list = ls())

####################################### Bootstrap Configuration ################################################
# Specify the bootstrap sample size, the accordingly OOB sample size, and the total number of bootstrap samples. 
B <- 200
# Taking a ratio value < 1 gives disturbance in the bootstrap sampling, 
# thus reducing the correlation among trees in the bagging forest. 
ratio <- 0.5

####################################### Implement Functions ####################################################
# Bootstrap Aggregation Algorithm.
Bagging_tree <- function(dat, B, ratio = 1) {
  # Train a bagging forest with out-of-bag misclassification rate.
  #
  # Args:
  #   dat: a data table, consisting of the dataset. 
  #   B: an integer variable, specifying the size of the bagging forest. 
  #   ratio: a numeric scalar between 0 and 1, 
  #          specifying the ratio of sample size for each bootstrap sample over the total size of the data. 
  # Returns:
  #   A list of three:
  #   1) forest: a list of models in the bagging forest;
  #   2) OOB_predicts: a numeric vector corresponding to the OOB prediction results for each observation in the dataset. 
  #   3) OOB_misclassification: a numeric scalar, indicating the OOB misclassification rate for the bagging forest. 
  Voting <- function(x) {
    # Compute the majority vote between class 1 and 2 for a vector. .
    #
    # Args:
    #   x: a vector, whose majority vote is to be computed.
    #   Note that the elements of x can only take value from 1, 2 and 0, 
    #   which indicates NULL. 
    # Returns:
    #   An integer, either 1 or 2, representing the majority vote result for vector x. 
    nums <- sapply(1 : length(unique(dat$y)), function(v) sum(x == v))
    res <- which.max(nums)
    return(res)
  }
  
  nobs <- dim(dat)[1]
  sample_size <- floor(nobs * ratio)

  # Get Bootstrap and OOB samples.  
  sample_index <- lapply(1 : B, function(i) sample(1 : nobs, size = sample_size, replace = TRUE))
  OOB_sample_index <- lapply(sample_index, function(s) setdiff(1 : nobs, s))
  
  samples_extracted <- lapply(list(sample_index, OOB_sample_index), 
                              function(samp) lapply(samp, function(s) dat[s, ]))
  samples <- samples_extracted[[1]]
  OOB_samples <- samples_extracted[[2]]
  
  # Train the models using the bootstrapped samples. 
  mods <- lapply(samples, function(s) rpart(y ~., data = s, method = "class"))
  
  # Evaluate the out-of-sample performance of the bagging forest using the OOB samples.
  # Group the prediction for OOB datapoints. 
  OOB_res <- Map(function(m, s) predict(m, s, type = "class"), mods, OOB_samples)
  OOB_mat <- matrix(0, nrow = nobs, ncol = B)
  for (i in 1 : B) {
    OOB_mat[OOB_sample_index[[i]], i] <- OOB_res[[i]]
  }
  # Get the OOB prediction for each observation of the dataset. 
  OOB_pred <- sapply(1 : nobs, function(i) Voting(OOB_mat[i, ]))
  # Calculate the OOB misclassification rate by comparison with the true classification results. 
  true_class <- as.numeric(dat[[ncol(dat)]])
  OOB_mis <- sum(OOB_pred != true_class) / length(true_class)
  return(list(forest = mods, OOB_predicts = OOB_pred, OOB_misclassification = OOB_mis))
}

####################################### Real Example ###########################################################
####################################### Configuration Data #####################################################
# Banknote authentication.
dat_banknote <- fread("C:/Users/Michael Zeng/Desktop/data_banknote_authentication.csv", header = F, stringsAsFactors = T)
dat <- dat_banknote
y_index = 5
dat[[y_index]] <- dat[[y_index]] + 1
mp <- ncol(dat) - 1
ms <- 2

# Sonor: classification of metal and rock. 
dat_sonar <- fread("C:/Users/Michael Zeng/Desktop/sonar.all-data.csv", header = F, stringsAsFactors = T)
dat <- dat_sonar
y_index = 61
mp <- ncol(dat) - 1
ms <- 2

# Breast cancer data: classification of benign and malignant. 
data(BreastCancer)
dat_BreastCancer <- BreastCancer[, -1]
dat <- as.data.table(dat_BreastCancer)
y_index = 10
mp <- ncol(dat) - 1
ms <- 2

# Iris data: . 
data(iris)
dat_iris <- iris
dat <- as.data.table(dat_iris)
y_index = 5
mp <- ncol(dat) - 1
ms <- 2

####################################### Data Preprocessing #####################################################
set.seed(1)
# Generic code.
dat <- na.omit(dat)
samp <- sample(1 : dim(dat)[1], dim(dat)[1] / 2)
# Data preprocessing. 
# Note that in code, each variable has to be coerce to numeric in order for comparison. 
dat_new_list <- lapply(dat, function(x) as.numeric(x))
dat_new <- as.data.table(Reduce(cbind, dat_new_list))
colnames(dat_new) <- names(dat_new_list)
# Rename the target variable as Y. 
names(dat_new)[y_index] <- "y"
dat <- dat_new
# Note: the final prepared dataset dat will have target variable y encoded as either 1 or 2. 

####################################### Model Training and Testing#### #########################################
bagging_trees <- Bagging_tree(dat, B, ratio)
bagging_trees$OOB_misclassification













