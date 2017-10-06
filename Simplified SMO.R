# Binary Classification by SVM Trained by a Simplified SMO Algorithm.
# Author: Michael Ruochen Zeng, micalzrc@gmail.com
library(mlbench)
library(data.table)

# The Simplified version of SMO simplifies the full SMO in the following manner:
# Instead of using some heuristic methods to find the optimal alpha_1 and alpha_2 in each iteration,
# we simply iterate over all alphas and pick alpha_1 randomly from those alphas that violate the KKT conditions,
# then alpha_2 is randomly selected from the remaining alphas. 
# The algo terminates if none of the alphas provide sufficient significant improvement after a few iterations. 
##########################################################
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

# Implement the simplified SMO algo. 
SimpleSMO <- function(data, target, C = 1, tol = 1E-3, maxPass = 200, ker = kernelOption) {
  # Train a SVM classifer using a simplified SMO algo.
  # Args:
  #   1) data: an object of type "data.table", containing the dataset.
  #   2) target: an integer, specifying the column representing the target variable. 
  #   3) C: a numeric variable, specifying the cost parameter of the SVM.
  #   4) tol: a numeric variable, specifying the tolerance level of the convergence. 
  #   5) maxPass: an integer variable, specifying the maximum number of passes allowed,
  #               if alpha is not further updated.
  #   6) kernelOption: a vector corresponding to either 'linear' or 'rbf'; 
  #                    if 'rbf', the 2nd element should be specified for sigma. 
  # Return:
  #   A list representing the SVM classifier. 
  # Extract y and x_mat as a vector and a matrix, respectively. 
  y <- data[, target, with = FALSE]
  y <- data.matrix(y)
  colnames(y) <- NULL
  y <- as.vector(y)

  x_mat <- data[, -target, with = FALSE]
  x_mat <- data.matrix(x_mat)
  colnames(x_mat) <- NULL
  
  # Initialization.
  m <- length(y)
  alpha <- rep(0, m)
  b <- 0
  
  # Compute the Gram matrix of the specified kernel using x_mat. 
  K <- CalcKernelMatrix(x_mat, ker)
  colnames(K) <- NULL
  
  # Implement the SVM classifier. 
  # Note that alphas, b in the enclosing environment of f, 
  # when they are updated in the execution environment of SimpleSMO
  f <- function(j) {
    return(sum(alpha * y * K[, j]) + b)
  }

  passes <- 0
  while (passes < maxPass) {
    num_alphas_changed <- 0
    i <- 0
    while (i < m) {
      i <- i + 1
      print(i)
      alpha_1 <- alpha[i]
      x_1 <- x_mat[i ,]
      y_1 <- y[i]
      
      E_1 <- f(i) - y_1
      r <- y_1 * E_1
      # If alpha_1 violates the KKT condition:
      if ((r < - tol && alpha_1 < C) || (r > tol && alpha_1 > 0)) {
        # Randomly select a j != i. 
        j <- sample(x = (1 : m)[- i], size = 1)
        x_2 <- x_mat[j]
        y_2 <- y[j]
        E_2 <- f(j) - y_2
        alpha_2 <- alpha[j]
        
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
          next
        }
        eta <- 2 * K[i, j] - K[i, i] - K[j, j]
        if (eta >= 0) {
          next
        }
        # Update alpha_2 and clip it if it lies outside (L, H).
        alpha[j] <- alpha_2 - y_2 * (E_1 - E_2) / eta
        alpha[j] <- AlphaClip(alpha[j], L, H)
        alpha2_step <- alpha_2 - alpha[j] 
        # Check if the step of which alpha_2 moves is siginificant. 
        if (abs(alpha2_step) < 1E-5) {
          next
        }
        alpha[i] <- alpha_1 + y_1 * y_2 *(alpha2_step)
        alpha1_step <-  alpha_1 - alpha[i]
        # Update b.
        b1 <- b - E_1 + y_1 * alpha1_step * K[i, i] + y_2 * alpha2_step * K[i, j]
        b2 <- b - E_2 + y_1 * alpha1_step * K[i, j] + y_2 * alpha2_step * K[j, j]
        if ((alpha[i] > 0) & (alpha[i] < C)) {
          b <- b1
        } else if ((alpha[j] > 0) & (alpha[j] < C)) {
          b <- b2
        } else {
          b <- (b1 + b2) / 2
        }
        
        num_alphas_changed <- num_alphas_changed + 1
      }
    }
    if (num_alphas_changed == 0) {
      passes <- passes + 1
    } 
  }
  return(list(weight = alpha, bias = b))#, kernel = K))
}

##########################################################
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

##########################################################
# kernelOption <- list('rbf', 1)
kernelOption <- list('linear')
res <- SimpleSMO(data = dat, target = y_index, C = 10)
plot(res$weight)
(sv_num <- sum(res$weight != 0))




