# Classification / Regression by multiple layers DNN using batch learning. 
# Author: Michael Ruochen Zeng, micalzrc@gmail.com
rm(list = ls())
library(pryr)
library(mlbench)

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
ANN_batch <- function(x_index = 1 : 4, y_index = 5, traindata = dat[samp, ], 
                           testdata = dat[- samp, ],
                           model = NULL,
                           # Specify the number of neurons in the hidden layer. 
                           hiddens = c(6), 
                           learning_rate = 0.01,
                           lambda = 1e-3,
                           random_seed = 1,
                           # Display the value of total loss every 'display' step.
                           display = 100,
                           max_iteration = 5000, 
                           threshold_convergence = 1e-2) { 
  # Define the prediction function. 
  Predict_ANN_batch <- function(m, X) {
    wm <- m$wm
    bm <- m$bm

    Us <- list()
    Zs <- list()
    Zs[[1]] <- X
    for (v in 1 : length(hiddens)) {
      Us[[v]] <- wm[[v]] %*% Zs[[v]] + bm[[v]]
      Zs[[v + 1]] <- matrix(sapply(as.vector(Us[[v]]), f), nrow = dims[v + 1])
    }
    
    U <- Us[[1]]
    V <- wm[[length(wm)]] %*% Zs[[length(Zs)]] + bm[[length(bm)]]
    Y_tildes <- lapply(1 : ncol(V), function(i) matrix(sapply(V[, i], partial(g, v = V[, i])), ncol = 1))
    Y_tilde <- Reduce(cbind, Y_tildes)
    labels_pred <- max.col(t(Y_tilde))
    
    return(labels_pred)
  }
  Delta_A <- function(x, y) {
    dE_i <- dE(x, y)# / n
    dg_i <- dg(as.vector(y))
    d_i <- diag(dE_i %*% dg_i)
  }
  set.seed(random_seed)  
  # Specify the total number of training data. 
  n <- nrow(traindata)
  
  # Extract the data and label.
  X <- t(unname(data.matrix(traindata[, x_index])))
  
  # Set the target variable for binary classification. 
  target <- traindata[, y_index]
  # Number of input nodes. 
  r <- nrow(X)
  
  if(is.factor(target)) { 
    # Create class labels for Y. 
    category_set <- sort(unique(target))
    # Number of output nodes. 
    s <- length(category_set)
    Y_m <- matrix(0, nrow = n, ncol = s)
    for (i in 1 : n) {
      Y_m[i, ][as.integer(target)[i]] <- 1
    } 
    Y_m <- t(Y_m) 
  } else {
    Y <- t(target)
    s <- dim(Y)[2]
  }
  
  # Initialize weight matrices for hidden layers.
  wm <- list()
  bm <- list()
  length(wm) <- length(hiddens) + 1
  length(bm) <- length(hiddens) + 1
  dims <- c(r, hiddens, s)
  for (i in 1 : length(wm)) {
    W <- 0.01 * matrix(rnorm(dims[i + 1] * dims[i]), nrow = dims[i + 1], ncol = dims[i])
    B <- matrix(0, nrow = dims[i + 1], ncol = n)
    wm[[i]] <- W
    bm[[i]] <- B
  }
  
  # Training the network.
  loss_total <- 1e6
  i <- 0
  Us <- list()
  Zs <- list()
  Zs[[1]] <- X
  while (i < max_iteration && loss_total > threshold_convergence) {  
#    while (i < 301) {  
    i <- i + 1
    
    # Forward feed. 
    for (v in 1 : length(hiddens)) {
      Us[[v]] <- wm[[v]] %*% Zs[[v]] + bm[[v]]
      Zs[[v + 1]] <- matrix(sapply(as.vector(Us[[v]]), f), nrow = dims[v + 1])
    }
    
    U <- Us[[1]]
    V <- wm[[length(wm)]] %*% Zs[[length(Zs)]] + bm[[length(bm)]]
    Y_tildes <- lapply(1 : ncol(V), function(i) matrix(sapply(V[, i], partial(g, v = V[, i])), ncol = 1))
    Y_tilde <- Reduce(cbind, Y_tildes)
    
    # Compute the loss E(w). 
    loss <- sum(sapply(1 : ncol(Y_tilde), function(i) E(Y_m[, i], Y_tilde[, i]))) / n
    #regularization  <- 0.5 * lambda * (sum(B * B) + sum(A * A))
    loss_total <- loss #+ regularization
    
    # Examine the evolution of the network for each epoch.
    if (i %% display == 0) {
      if (is.factor(target)) {
        model <- list(wm = wm, bm = bm)  
       # class_label <- PredictANN_batch(m = model, t(testdata[, x_index]))
      #  accuracy <- mean(testdata[, y_index] == category_set[class_label])
        
        class_label <- Predict_ANN_batch(m = model, X = t(unname(data.matrix(testdata[, x_index]))))
        accuracy <- mean(as.factor(testdata[, y_index]) == category_set[class_label])
        cat(i, loss_total, accuracy, "\n")
      } else {
        cat(i, loss_total, "\n")
      }
    }
    
    # Backpropagation. 
    # From output layer to the hidden layer. 
    # Corresponds to (10.37) of Izenman(2008). 
    d_list <- lapply(1 : ncol(Y_tilde), function(i) Delta_A(Y_m[, i], Y_tilde[, i]))
    d <- Reduce(cbind, d_list) / n
    # Corresponds to (10.49) of Izenman(2008).  
    dwm <- list()
    dbm <- list()
    dfs <- list()
    ds <- list()
    num <- length(hiddens) + 1
    
    ds[[num]] <- d
    dwm[[num]] <-  ds[[num]] %*% t(Zs[[num]]) 
    dbm[[num]] <- matrix(rep(rowSums(d), n), nrow = s)
    for (v in length(hiddens) : 1) { 
      # From hidden layer to the input layer. 
      # Corresponds to f_j'(U_ij) in (10.47) of Izenman(2008). 
      dfs[[v]] <- df(Us[[v]]) 
      # Corresponds to (10.47) of Izenman(2008).
      ds[[v]] <- matrix(as.vector(t(wm[[v + 1]]) %*% ds[[v + 1]]) * as.vector(dfs[[v]] ), nrow = hiddens[v])
      # Corresponds to (10.50) of Izenman(2008).
      dwm[[v]] <-  ds[[v]] %*% t(Zs[[v]])
      dbm[[v]] <- matrix(rep(rowSums(ds[[v]]), n), nrow = hiddens[[v]])
    }
    
    # Update the weights. 
    #dB <- dB + lambda * B
    #A <- dA  + lambda * A
    for (u in 1 : length(wm)) {
      wm[[u]] <- wm[[u]] - learning_rate * dwm[[u]]
      bm[[u]] <- bm[[u]] - learning_rate * dbm[[u]]
    }  
  }
  model <- list(wm, bm)    
  return(model)
}

############################################ Testing #########################################################
set.seed(1)

#dat <- iris[1 : 100, ]
dat <- iris
#data("BreastCancer")
#dat <- BreastCancer
dat <- na.omit(dat)
samp <- sample(1 : dim(dat)[1], dim(dat)[1] / 2)
train_dat <- dat[samp, ]
test_dat <- dat[- samp, ]
if (nrow(train_dat) > nrow(test_dat)) {
  train_dat <- train_dat[- 1 ,]
} else if (nrow(train_dat) < nrow(test_dat)) {
  test_dat <- test_dat[- 1 ,]
}

mod <- ANN_batch(x_index = 1 : 4, y_index = 5, traindata = train_dat, 
                 testdata = test_dat, hiddens = c(6, 10), max_iteration = 2500,
                 learning_rate = 0.05, lambda = 1e-3, display = 100, 
                 random_seed = 1, threshold_convergence = 1e-2)

