# Multiclassification by multiple hidden layers ANN using batch learning with cross-entropy loss. 
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

Logit <- function(x) {
  1 / (1 + exp(-x))
}

dLogit <- function(x) {
  Logit(x) * (1 - Logit(x))
}

SoftMax <- function(x, k) {
  exp(x[k]) / sum(exp(x))
}

############################################ Configuration ####################################################
f <- ReLu 
df <- dReLu
g <- SoftMax

############################################ Implement Training Function ######################################
Multiclass_ANN_batch <- function(x_index = 1 : 4, y_index = 5, traindata = iris[samp, ], 
                           testdata = iris[- samp, ],
                           model = NULL,
                           # Specify the number of neurons in the hidden layer. 
                           hiddens = c(6, 10), 
                           learning_rate = 0.1,
                           lambda = 1e-3,
                           random_seed = 1,
                           # Display the value of total loss every 'display' step.
                           display = 100,
                           max_iteration = 5000, 
                           threshold_convergence = 1e-2) { 
  # Define the prediction function. 
  Pred_Multiclass_ANN_batch <- function(m, X) {
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
    Y_tilde <- lapply(1 : n, function(j) sapply(1 : s, partial(g, x = V[, j])))
    Y_tilde <- Reduce(cbind, Y_tilde)
    labels_pred <- max.col(t(Y_tilde))
    
    return(labels_pred)
  }
  set.seed(random_seed)  
  # Specify the total number of training data. 
  n <- nrow(traindata)
  
  # Extract the data and label.
  X <- t(unname(data.matrix(traindata[, x_index])))
  
  # Set the target variable for binary classification. 
  target <- traindata[, y_index]
  target <- as.factor(target)
  category_set <- sort(unique(target))
  
  # Get the number of input nodes.
  r <- nrow(X)
  # Get the number of output nodes. 
  s <- length(category_set)
  
  # Create matrix Y_m.   
  Y_m <- matrix(0, nrow = n, ncol = s)
  for (i in 1 : n) {
    Y_m[i, ][as.integer(target)[i]] <- 1
  } 
  Y_m <- t(Y_m)
  
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
    i <- i + 1
    
    # Forward feed. 
    for (v in 1 : length(hiddens)) {
      Us[[v]] <- wm[[v]] %*% Zs[[v]] + bm[[v]]
      Zs[[v + 1]] <- matrix(sapply(as.vector(Us[[v]]), f), nrow = dims[v + 1])
    }
    
    U <- Us[[1]]
    V <- wm[[length(wm)]] %*% Zs[[length(Zs)]] + bm[[length(bm)]]
    Y_tilde <- lapply(1 : n, function(j) sapply(1 : s, partial(g, x = V[, j])))
    Y_tilde <- Reduce(cbind, Y_tilde)
    
    # Compute the loss E(w). 
    loss <- sum(diag(t(Y_m) %*% (- log(Y_tilde)))) / n
    #reg_loss  <- 0.5 * reg * (sum(B * B) + sum(A * A))
    loss_total <- loss #+ reg_loss
  
    # Examine the evolution of the network for each epoch.
    if(i %% display == 0) {
      if (is.factor(target)) {
        model <- list(wm = wm, bm = bm)  
        class_label <- Pred_Multiclass_ANN_batch(m = model, X = t(unname(data.matrix(testdata[, x_index]))))
        accuracy <- mean(as.factor(testdata[, y_index]) == category_set[class_label])
        cat(i, loss_total, accuracy, "\n")
      } else {
        cat(i, loss_total, "\n")
      }
    }
  
    # Backpropagation. 
    # From output layer to the hidden layer. 
    # Corresponds to (10.37) of Izenman(2008). 
    d <- Y_tilde - Y_m
    d <- d / n 
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
    #dB <- dB + reg * B
    #A <- dA  + reg * A
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

mod <- Multiclass_ANN_batch(x_index = 1 : 4, y_index = 5, traindata = train_dat, 
                            testdata = test_dat, hiddens = c(6, 10), max_iteration = 2500,
                            learning_rate = 0.05, lambda = 1e-3, display = 100, 
                            random_seed = 1, threshold_convergence = 1e-2)

