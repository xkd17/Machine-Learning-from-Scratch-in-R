# Multiclassification by Bagging Decision Tree Models.
# Author: Michael Ruochen Zeng, micalzrc@gmail.com
library(rpart)
library(pryr)
library(mlbench)
library(data.table)
rm(list = ls())

####################################### Configuration Data #####################################################
# Banknote authentication.
dat_banknote <- fread("C:/Users/mzeng/Desktop/data_banknote_authentication.csv", header = F, stringsAsFactors = T)
dat <- dat_banknote
y_index = 5
dat[[y_index]] <- dat[[y_index]] + 1

# Sonor: classification of metal and rock. 
dat_sonar <- fread("C:/Users/mzeng/Desktop/sonar.all-data.csv", header = F, stringsAsFactors = T)
dat <- dat_sonar
y_index = 61

# Breast cancer data: classification of benign and malignant. 
data(BreastCancer)
dat_BreastCancer <- BreastCancer[, -1]
dat <- as.data.table(dat_BreastCancer)
y_index = 10

# Iris data: 
data(iris)
dat_iris <- iris
dat <- as.data.table(dat_iris)
y_index = 5

####################################### Data Preprocessing #####################################################
#set.seed(1)
# Generic code.
mp <- ncol(dat) - 1
ms <- 2

dat <- na.omit(dat)
# Data preprocessing. 
# Note that in code, each variable has to be coerce to numeric in order for comparison. 
dat_new_list <- lapply(dat, function(x) as.numeric(x))
dat_new <- as.data.table(Reduce(cbind, dat_new_list))
colnames(dat_new) <- names(dat_new_list)
# Rename the target variable as Y. 
names(dat_new)[y_index] <- "y"
dat <- dat_new
# Note: the final prepared dataset dat will have target variable y encoded as either 1 or 2. 

####################################### Implement Boosting #####################################################
# Binary classification: the target variable y should be either 1 or -1. 
target <- dat$y
y <- ifelse(target == 1, -1, 1)
dat$y <- y

# Data partition. 
samp <- sample(1 : dim(dat)[1], dim(dat)[1] / 2)
train_dat <- dat[samp, ]
test_dat <- dat[- samp, ]
if (nrow(train_dat) > nrow(test_dat)) {
  train_dat <- train_dat[-1 ,]
} else if (nrow(train_dat) < nrow(test_dat)) {
  test_dat <- test_dat[-1 ,]
}

# Train the boosting tree. 
T <- 10
nobs <- nrow(train_dat)
# Initialize the weights. 
w <- rep(1 / nobs, length = nobs)
mods <- list()
betas <- list()
for (t in 1 : T) {
  mod <- rpart(y ~., data = train_dat, weights = w, 
               method = "class", control = rpart.control(maxdepth = 1))
  pred <- as.numeric(as.character(predict(mod, data = train_dat, type = "class")))
  e <- as.integer(pred != train_dat$y)
  PE <- sum(w * e) / sum(w)
  beta <- log((1 - PE) / PE) / 2
  w <- w * exp(2 * beta * e)
  w <- w / sum(w)
  mods[[t]] <- mod
  betas[[t]] <- beta
}
preds <- lapply(mods, function(m) as.numeric(as.character(predict(m, data = test_dat, type = "class"))))
preds_m <- Reduce(cbind, preds)
OOS_pred <- sapply(1 : nrow(preds_m), function(i) sum(preds_m[i, ] * unlist(betas)))
OOS_pred <- ifelse(OOS_pred > 0, 1, -1)
#preds_weighted <- Reduce(cbind, Map(function(p, w) p * w, preds, betas))
#OOS_pred <- ifelse(rowSums(preds_weighted) > 0, 1, -1)
(misclass_rate <- sum(OOS_pred != test_dat$y) / length(OOS_pred))




