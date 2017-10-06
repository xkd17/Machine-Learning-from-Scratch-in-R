# Binary Classification by SVM Trained by SMO Algorithm.
# Author: Michael Ruochen Zeng, micalzrc@gmail.com
########################################################
library(data.table)
library(mlbench)
library(pryr)
library(MASS)

########################################################
# X: a matrix to hold the training dataset.
# y: class label vector. 
# C: cost parameter.
# alpha: a vector to hold the lagrange multipliers. 
# b: a scalar representing the bias term. 
# errors: error cache. 
# m: an integer specifying the size of the training dataset. 
# K: the Gram matrix of the kernel function evaluated using X. 

setClass("svm", representation(X = "matrix", y = "numeric", cost = "numeric",
                               m = "integer", alpha = "numeric",
                               b = "numeric", error = "numeric", K = "matrix", 
                               obj_value = "ANY"))

# Note that errorCache has two columns:
# 1) the ith row in the 1st column indicates the position and status (zero or not) of alpha_i;
# 2) the ith row in the 2nd column refers to error_i.

########################################################
# Calulate kernel value.  
CalcKernelValue <- function(matrix_x, sample_x, kernelOption) {
  # Compute the value of kernel function K(x_i, x).
  # Args:
  #   1) matrix_x: a matrix, whose ith rows represents x_i.
  #   2) sample_x: a numeric vector, representing a specific row of matrix_x. 
  #   3) kernelOption: a list corresponding to either 'linear' or 'rbf'; 
  #                    if 'rbf', the 2nd element should be specified for sigma. 
  # Return:
  #   A numeric vector storing the computed kernel value for x_i and x, i = 1, 2,..., n.
  kernelType = kernelOption[[1]] 
  
  if (kernelType == 'linear') {
    kernelValue <- apply(t(t(matrix_x) * sample_x), 1, sum)
    names(kernelValue) <- NULL
  } else if (kernelType == 'rbf') {
    sigma = kernelOption[[2]] 
    # If sigma is 0, an error will be resulted in, so transform it to the default value 1. 
    if (sigma == 0) {
      sigma <- 1
    }
    # Create a matrix whose ith row is the difference between x_i and x. 
    diff <- t(t(matrix_x) - sample_x)
    kernelValue <- sapply(1 : nrow(diff), function(i) exp(sum(diff[i, ] ^ 2) / (- 2 * sigma ^ 2)))
  } else {
    return("Error: not supported kernel type!")
  }
  return(kernelValue)  
}

# Calculate kernel matrix given the training set and kernel type.  
CalcKernelMatrix <- function(train_x, kernelOption) {
  # Calculate the Gram matrix for a dataset stored as a matrix using specified kernel type.
  # Args:
  #   1) train_x: a matrix whose ith rows represents the ith observation x_i.
  #   2) kernelOption: a vector corresponding to either 'linear' or 'rbf'; 
  #                    if 'rbf', the 2nd element should be specified for sigma. 
  # Return:
  #   A numeric matrix whose (i, j)th element is K(x_i, x_j), 
  #   with x_i being the ith row of the matrix train_x. 
  kernelMatrix <- Reduce(cbind, 
                         lapply(1 : nrow(train_x), 
                                function(i) CalcKernelValue(matrix_x = train_x, 
                                                            sample_x = train_x[i, ], 
                                                            kernelOption)))
  return(kernelMatrix) 
}

# Clip the a value if it lies outside a specified range defined by the lower and upper bounds, l & h. 
AlphaClip <- function(a, l, h) {
  # Clip an input value,
  # if it lies outside a specified range defined by the lower and upper bounds, l & h.
  # Args:
  #   1) a: a numeric value to be clipped.
  #   2) l: a numeric value specifying the lower bound. 
  #   3) h: a numeric value specifying the upper bound.
  # Return:
  #   A numeric value, could be l, h or a itself.
  if (a < l) {
    a <- l
  } else if (a > h) {
    a <- h
  } 
  return(a)
}

# Implement the SVM classifier. 
SVM_classifier <- function(j, mod) {
  # Compute the result for a svm classifier based on a svm model and a position index.
  # Args:
  #   1) j: an integer, indicating the position of observation in the dataset to be classified. 
  #   2) mod: an object of type "svm". 
  # Return:
  #   a numeric value, 
  #   f(x_j) =  sum over alpha_i * y_i * K(x_i, x_j) + b over all i.
  return(sum(mod@alpha * mod@y * mod@K[, j]) + mod@b)
}

# Compute the objective function of the dual problem,
# which is sum of alpha - 1/2 * v'Kv, 
# where the elements of v is the product of vector y & alpha at the corresponding position,
# and K is the Gram matrix of the kernel function evaluated at X. 
Obj_fun <- function(alpha, mod) {
  # Compute the value of the objective function of the dual problem of a SVM,
  # using an updated alpha vector.
  # Args:
  #   1) alpha: a numeric vector to hold the lagrange multipliers. 
  #   2) mod: an object of type "svm", whose alpha slot is to be updated. 
  # Return:
  #   a numeric value, 
  #   sum of alpha - 1/2 * v'Kv.
  v <- matrix(mod@y * alpha, ncol = 1)
  return(sum(alpha) - 0.5 * t(v) %*% mod@K %*% v)
}

########################################################
# Implement the three fundamental functions for training a SVM model using SMO. 
Take_step <- function(i, j, mod) {
  # Load items from mod.
  K <- mod@K
  alpha <- mod@alpha
  m <- mod@m
  x_mat <- mod@X
  C <- mod@cost
  y <- mod@y
  b <- mod@b
  
  # Skip the case when i = j. 
  if (i == j) {
    return(list(0, mod))
  }
  
  alpha_1 <- alpha[i]
  x_1 <- x_mat[i ,]
  y_1 <- y[i]
  # E_1 <- mod@error[i]
  E_1 <- SVM_classifier(i, mod) - y_1

  alpha_2 <- alpha[j]
  x_2 <- x_mat[j]
  y_2 <- y[j]
  E_2 <- mod@error[j]
  #E_2 <- SVM_classifier(j, mod) - y_2

  # Compute L and H.
  alpha_sum <- alpha_1 + alpha_2
  alpha_diff <- alpha_2 - alpha_1
  if (y_1 == y_2) {
    L <- max(0, alpha_sum - C)
    H <- min(C, alpha_sum)
  } else {
    L <- max(0, alpha_diff)
    H <- min(C, alpha_diff + C)
  }
  if (L == H) {
    return(list(0, mod))
  }
  
  # Compute the eta. 
  eta <- 2 * K[i, j] - K[i, i] - K[j, j]
  # Update alpha_2 and clip it if it lies outside (L, H).
  if (eta >= 0) {
    alpha_temp <- alpha
    
    alpha_temp[j] <- L
    Lobj <- Obj_fun(alpha_temp, mod)
    
    alpha_temp[j] <- H
    Hobj <- Obj_fun(alpha_temp, mod)
    
    if (Lobj > Hobj + eps) {
      alpha[j] <- L
    } else if (Lobj > Hobj - eps) {
      alpha[j] <- H
    } else {
      alpha[j] <- alpha_2
    }
    
  } else {
    alpha[j] <- alpha_2 - y_2 * (E_1 - E_2) / eta
    alpha[j] <- AlphaClip(alpha[j], L, H)
  }
  alpha2_step <- alpha[j] - alpha_2
  
  # Push alpha_2 to 0 or C if very close.
  if (alpha[j] < 1E-5) {
    alpha[j] <- 0
  } else if (alpha[j] > C - 1E-5) {
    alpha[j] <- C
  }

  # Check if the step of which alpha_2 moves is siginificant. 
  if (abs(alpha2_step) < eps * (alpha[j] + alpha_2 + eps)) {
    return(list(0, mod))
  }
  
  # Update alpha_1. 
  alpha[i] <- alpha_1 - y_1 * y_2 * (alpha2_step)
  alpha1_step <-  alpha[i] - alpha_1
  
  # Update b.
  b1 <- b - E_1 - y_1 * alpha1_step * K[i, i] - y_2 * alpha2_step * K[i, j]
  b2 <- b - E_2 - y_1 * alpha1_step * K[i, j] - y_2 * alpha2_step * K[j, j]
  if ((alpha[i] > 0) & (alpha[i] < C)) {
    b <- b1
  } else if ((alpha[j] > 0) & (alpha[j] < C)) {
    b <- b2
  } else {
    b <- (b1 + b2) / 2
  }
  
  # Update mod. 
  mod@alpha <- alpha
  mod@b <- b
  # print(c(alpha[i], alpha[j], b))
  
  # Update error cache:
  # 1) for optimized alphas i and j, error is set to 0 if they are in the unbound set; 
  Update_error <- function(i) {
    if ((alpha[i] > 0) && (alpha[i] < C)) {
      return(0)
    } else {
      return(SVM_classifier(i, mod) - y[i])
    }
  }
  mod@error[c(i, j)] <- sapply(c(i, j), Update_error)
  
  # 2) for non-optimized erros, update them using f(x_i) - y_i.
  # Note that the mod has already updated, so are alpha and b.
  np_index <- (1 : m)[- c(i, j)]
  mod@error[np_index] <- sapply(np_index, function(p) SVM_classifier(p, mod) - y[p]) 
  
  return(list(1, mod))
}

Exmaine_example <- function(j, mod) {
  y_2 <- mod@y[j]
  alpha_2 <- mod@alpha[j]
  E_2 <- mod@error[j]
  #E_2 <- SVM_classifier(j, mod) - y_2
  #print(c(E, E_2, E - E_2))
  r <- y_2 * E_2
  
  # The outer loop first iterates over the entire sample, 
  # and then over all non-bound examples.
  # During the outer loop, it checks whether each example violates the KKT conditions. 
  # Check this jth point of the dataset violates the KKT condition or not,
  # and those violating examples are eligible for optimization procedure. 
  if ((r < - tol && alpha_2 < mod@cost) || (r > tol && alpha_2 > 0)) {
    # Now go through the herarchy of second choice heuristics. 
    # Use the 1st heuristic:maximize the step size taken during joint optimization, i.e. |E_2 - E_1|.
    # If the number of nonzero & non-C alphas is greater than 1:
    if (sum((mod@alpha != 0) * (mod@alpha != mod@cost)) > 1) {
      if (E_2 > 0) {
        i <- which.min(mod@error)
      } else {
        i <- which.max(mod@error)
      }
      res <- Take_step(i, j, mod)
      mod <- res[[2]]
      if (res[[1]]) {
        return(res)
      }
    }
    # Use the 2nd heuristic: iterating through the non-bound examples, 
    # searching for an second example that can make positive progress. 
    non_bound_index <- which(mod@alpha != 0 & mod@alpha != mod@cost)
    rand_sampl <- sample(x = non_bound_index, size = length(non_bound_index))
    for (i in rand_sampl) {
      res <- Take_step(i, j, mod)
      mod <- res[[2]]
      if (res[[1]]) {
        return(res)
      }
    }
    # Use the 3rd heuristic: iterating through entire training dataset, 
    # until an second example is found to make positive progress. 
    for (i in sample(1 : mod@m, size = mod@m)) {
      res <- Take_step(i, j, mod)
      mod <- res[[2]]
      if (res[[1]]) {
        return(res)
      }
    }
  }
  return(list(0, mod))
}

Train_svm <- function(mod) {
  numChanged <- 0
  examineAll <- 1
  
  while ((numChanged > 0) || (examineAll)) {
    numChanged <- 0
    # The outer loop first iterates over the entire training set. 
    if (examineAll) {
      for (i in 1 : mod@m){
        res <- Exmaine_example(i, mod)
        mod <- res[[2]]
        if (res[[1]]) {
          obj_value <- Obj_fun(mod@alpha, mod)
          mod@obj_value <- append(mod@obj_value, obj_value)
        }
        
        numChanged <- numChanged + res[[1]]
      }
    # After one pass through the entire training set, 
    # the outer loop iterates over all examples whose Lagrange multipliers are neither 0 nor C. 
    } else {
      for (i in which(mod@alpha != 0 & mod@alpha != mod@cost)) {
        res <- Exmaine_example(i, mod)
        mod <- res[[2]]
        if (res[[1]]) {
          obj_value <- Obj_fun(mod@alpha, mod)
          mod@obj_value <- append(mod@obj_value, obj_value)
        }
        
        numChanged <- numChanged + res[[1]]
      }
    }
    # The outer loop keeps alternating
    # between single passes over the single pass over entire traiing set 
    # and multiple passes over the non-bound subsets,
    # until the entire training set obeys the KKT conditions within the eps, 
    # whereupon the algo terminates. 
    if (examineAll == 1) {
      examineAll <- 0
    } else if (numChanged == 0) {
      examineAll <- 1
    }
    # The while loop will terminate if examineAll = 0 but numChanged = 0,
    # in which case, the model mod has not been updated. 
    # This could happend when initially examineAll = 0 and numChanged > 0, 
    # and numChanged reduces to 0 after multiple passes over the non-bound subsets.
    print(numChanged)
  }
  temp <- mod@alpha
  temp[temp < 0] <- 0
  mod@alpha <- temp
  return(mod)
}

###########################################################
# Iris data: classification of flower species. 
dat <- iris[1 : 100, -5]
y <- as.numeric(iris[1 : 100, ]$Species)
dat <- as.data.table(cbind(dat, y))
colnames(dat)[5] <- "y"
y_index = 5
dat[[y_index]] <- as.numeric(dat[[y_index]])
dat[dat[[y_index]] == 2, ][[y_index]] <- -1

# Breast cancer data: classification of benign and malignant. 
data(BreastCancer)
dat_BreastCancer <- na.omit(BreastCancer[, -1])
dat <- as.data.table(dat_BreastCancer)
y_index = 10
dat[[y_index]] <- as.numeric(dat[[y_index]])
dat[dat[[y_index]] == 2, ][[y_index]] <- -1

# Sonor: classification of metal and rock. 
dat_sonar <- fread("C:/Users/Zeng Ruochen/Desktop/sonar.all-data.csv", header = F, stringsAsFactors = T)
dat <- dat_sonar
y_index = 61
dat[[y_index]] <- as.numeric(dat[[y_index]])
dat[dat[[y_index]] == 2, ][[y_index]] <- -1

# Banknote authentication.
dat_banknote <- fread("C:/Users/Zeng Ruochen/Desktop/data_banknote_authentication.csv", header = F, stringsAsFactors = T)
dat <- dat_banknote
y_index = 5
dat[[y_index]] <- as.numeric(dat[[y_index]])
dat[dat[[y_index]] == 0, ][[y_index]] <- -1

###########################################################
# Define a SVM class for storing variables and data.
m <- nrow(dat)
ker <- list("rbf", 3)
kernel <- CalcKernelMatrix(data.matrix(dat)[, 1 : (y_index - 1)], ker)
colnames(kernel) <- NULL
alpha_init <- rep(0, m)
C <- 1

svm_instance <- new("svm", X = data.matrix(dat)[, 1 : (y_index - 1)], 
                    y = data.matrix(dat)[, y_index], cost = C, 
                    m = m, alpha = alpha_init, b = 0, error = rep(0, m), K = kernel, 
                    obj_value = c())
error_init <- sapply(1 : m, partial(SVM_classifier, mod = svm_instance)) - svm_instance@y
svm_instance@error <- error_init

###########################################################
# Set tolerances.
# Tolerance for the convergence on the error.  
tol <- 1E-3
# Tolerance for the convergence on alpha. 
eps <- 1E-3
res <- Train_svm(svm_instance)
plot(res@alpha)
(sv_num <- sum(res@alpha != 0))

# Plot the trajectory of the objective function. 
plot(res@obj_value)
abline(h = max(res@obj_value))

###########################################################
# Synthetic data. 
x <- matrix(rnorm (200 * 2), ncol = 2)
x[1 : 100 ,] <- x [1 : 100, ] + 3
x[101 : 150 ,] <- x [101 : 150, ] 
y <- c(rep(1, 110), rep (- 1, 90))
plot(x, col = 3 - y)
dat <- data.table(cbind(x, y))
y_index <- 3

# Set up a svm instance. 
m <- nrow(dat)
ker <- list("linear", 1)
kernel <- CalcKernelMatrix(data.matrix(dat)[, 1 : (y_index - 1)], ker)
colnames(kernel) <- NULL
alpha_init <- rep(0, m)
C <- 1

svm_instance <- new("svm", X = data.matrix(dat)[, 1 : (y_index - 1)], 
                    y = data.matrix(dat)[, y_index], cost = C, 
                    m = m, alpha = alpha_init, b = 0, error = rep(0, m), K = kernel, 
                    obj_value = c())
error_init <- sapply(1 : m, partial(SVM_classifier, mod = svm_instance)) - svm_instance@y
svm_instance@error <- error_init

# Set tolerances.
# Tolerance for the convergence on the error.  
tol <- 1E-3
# Tolerance for the convergence on alpha. 
eps <- 1E-3

# Train a SVM model. 
res <- Train_svm(svm_instance)
plot(res@alpha)
(sv_num <- sum(res@alpha != 0))
plot(res@obj_value)
abline(h = max(res@obj_value))

# Obtain the decision boundary. 
Decision_boundary <- function(mod){
  w <- rowSums(sapply(1 : m, function(i) (mod@alpha * mod@y)[i] * mod@X[i, ]))
  b <- mod@b
  
  function(x) {
    sum(w * x) + b
  }
}
f <- Decision_boundary(res)

# Plot the decision boundary.
w <- rowSums(sapply(1 : m, function(i) (res@alpha * res@y)[i] * res@X[i, ]))
min_x1 = min(res@X[, 1]) 
max_x1 = max(res@X[, 1])
min_x2 <- (- res@b - w[1] * min_x1) / w[2]
max_x2 <- (- res@b - w[1] * max_x1) / w[2]

# Plot the soft margin. 
min_margin1 <- (1 - res@b - w[1] * min_x1) / w[2]
max_margin1 <- (1 - res@b - w[1] * max_x1) / w[2]

min_margin2 <- (- 1 - res@b - w[1] * min_x1) / w[2]
max_margin2 <- (- 1 - res@b - w[1] * max_x1) / w[2]

# Plot the decision boundary, the soft margin and the support vectors. 
l <- 1000
plot(x = seq(min_x1, max_x1, length.out = l), y = seq(min_x2, max_x2, length.out = l), type = 'l')
lines(x = seq(min_x1, max_x1, length.out = l), 
      y = seq(min_margin2, max_margin2, length.out = l), col = "black", type = "l", lty = 2)
lines(x = seq(min_x1, max_x1, length.out = l), 
      y = seq(min_margin1, max_margin1, length.out = l), col = "black", type = "l", lty = 2)

points(x, col = 3 - y)
points(x[res@alpha > 0, ], col = 3 - y[res@alpha > 0], pch = 3)
res@alpha[res@alpha < 0]




