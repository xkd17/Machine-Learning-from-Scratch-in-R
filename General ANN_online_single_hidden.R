# Classification / Regression by single hidden layer DNN using online learning. 
# Author: Michael Ruochen Zeng, micalzrc@gmail.com
rm(list = ls())
library(pryr)

############################################ Implement Activation Functions ####################################
ReLu <- function(x) {
  max(0, x)
}

dReLu <- function(x) {
  ifelse(x > 0, 1, 0)
}

Logit <- function(x, v) {
  1 / (1 + exp(-x))
}

dLogit <- function(y) {
  diag(y * (1 - y))
}

SoftMax <- function(x, v) {
  exp(x) / sum(exp(v))
}

dSoftMax <- function(y) {
   s <- length(y)
   v <- rep(y, s)
   m <- matrix(v, nrow = s)
   - v * (t(m) - diag(s))
}

Cross_entropy <- function(x, y) {
  - (t(x) %*% log(y))
}

dCross_entropy <- function(x, y) {
  s <- length(x)
  - t(matrix(rep(x / y, s), ncol = s))
}

Square_Loss <- function(x, y) {
  sum((x - y) ^ 2) / 2
}

dSquare_loss <- function(x, y) {
  s <- length(x)
  -t(matrix(rep(x - y, s), nrow = s))
}

Linear <- function(x) {
  x
}

dLinear <- function(y) {
  diag(length(y))
}
############################################ Configuration ####################################################
f <- ReLu
df <- dReLu
g <- SoftMax
dg <- dSoftMax
E <- Cross_entropy
dE <- dCross_entropy

############################################ Implement Training Function ######################################
TrainANN_online <- function(x_index = 1 : 4, y_index = 5, 
                            traindata = dat, 
                            testdata = dat,
                            model = NULL,
                            t = 6, 
                            learning_rate = 0.05,
                            lambda = 1e-3,
                            random_seed = 1) {
  # Implement the function to compute delta values for matrix A. 
  Delta_A <- function(x, y) {
    dE_i <- dE(x, y)# / n
    dg_i <- dg(as.vector(y))
    d_i <- diag(dE_i %*% dg_i)
  }
  set.seed(random_seed)
  
  # Specify the total number of observations. 
  n <- nrow(traindata)
  
  # Extract the data and label.
  X <- t(unname(data.matrix(traindata[, x_index])))
  
  # Set the target variable for binary classification. 
  target <- traindata[, y_index]
  # Number of input nodes. 
  r <- nrow(X)
  # Generate the matrix for output nodes. 
  if(is.factor(target)) { 
    # Create class labels for Y. 
    category_set <- sort(unique(target))
    # Number of output nodes. 
    s <- length(category_set)
    Y1 <- as.integer(target) 
    if (length(category_set) == 2) {
      Y1[Y1 == unique(Y1)[1]] <- 1
      Y1[Y1 == unique(Y1)[2]] <- - 1
      Y2 <- 0 - Y1
      Y_m <- cbind(Y1, Y2)
      Y_m[Y_m == - 1] <- 0
    } else {
      Y_m <- matrix(0, nrow = n, ncol = s)
      for (i in 1 : n) {
        Y_m[i, ][Y1[i]] <- 1
      }
    }  
  } else {
    Y <- t(target)
    s <- dim(Y)[2]
  }
  
  # Initilize the weight and bias matrices. 
  B <- 0.01 * matrix(rnorm(t * r), nrow = t, ncol = r)
  b_0 <- matrix(0, nrow = t, ncol = 1)
  
  A <- 0.01 * matrix(rnorm(s * t), nrow = s, ncol = t)
  a_0 <- matrix(0, nrow = s, ncol = 1)
  model <- list(B = B, b_0 = b_0, A = A, a_0 = a_0)
  
  # Training the network
  i <- 0
  while(i < n) {    
    i <- i + 1
    # Consider the ith observation. 
    X_i <- X[, i]
    if (is.factor(target)) {
      Y_i <- Y_m[i, ]
      Y_i <- matrix(Y_i, ncol = 1)
    } else {
      Y_i <- Y[i]
    }
    
    # Forward feed. 
    U <- B %*% X_i + b_0
    Z <- matrix(sapply(U, f), ncol = 1)
    V <- A %*% Z + a_0
    Y_tilde <- matrix(sapply(V, partial(g, v = V)), ncol = 1)
    
    # Compute the loss.
    loss <- E(Y_i, Y_tilde) 
    regularization  <- 0.5 * lambda * (sum(B * B) + sum(A * A))
    loss_total <- loss + regularization
    
    # Display the loss and the prediction outcome (T / F). 
    if (is.factor(target)) {
      class_label <- which.max(Y_tilde)
      pred_res <- testdata[, y_index][i] == category_set[class_label]
      cat(i, loss_total, pred_res, "\n")
    } else {
      cat(i, loss_total, testdata[, y_index][i], "\n") 
    }  

    # Backpropagation. 
    # From output layer to the hidden layer. 
    # Corresponds to (10.37) of Izenman(2008). 
    d_i <- Delta_A(Y_i, Y_tilde)
    # Corresponds to (10.36) of Izenman(2008). 
    dA <-  d_i %*% t(Z) 
    da_0 <- d_i
    
    # From hidden layer to the input layer. 
    # Corresponds to f_j'(U_ij) in (10.47) of Izenman(2008). 
    df_U <- df(U) 
    # Corresponds to (10.47) of Izenman(2008).
    d2_i <- as.vector(t(d_i) %*% A) * as.vector(df_U) 
    d2_i <- matrix(d2_i, ncol = 1) 
    # Corresponds to (10.48) of Izenman(2008).
    dB <-  d2_i %*% matrix(X_i, nrow = 1) 
    db_0 <-  d2_i
    
    # Update the weights.
    dB <- dB + lambda * B
    dA <- dA  + lambda * A
    
    B <- B - learning_rate * dB
    b_0 <- b_0 - learning_rate * db_0
    
    A <- A - learning_rate * dA
    a_0 <- a_0 - learning_rate * da_0    
  }
  model <- list(B = B, b_0 = b_0, A= A, a_0 = a_0)    
  return(model)
}

############################################ Testing #########################################################
#dat <- iris[1 : 100, ]
dat <- iris
ir_model <- TrainANN_online(x_index = 1 : 4, y_index = 5, traindata = dat, 
                            testdata = dat, t = 10, learning_rate = 0.5)

