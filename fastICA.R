# Implementation of FastICA with Parallel Decorrelation algorithm 
# to perform Independent Component Analysis (ICA),
# using logcosh function to approximate the negentropy. 

# Author: Ruochen Zeng, Department of Statistics and Actuarial Science, University of Hong Kong
# Email: xkd17@hku.hk
# Date: 18/10/2017
# Reference: 
# 1) Aapo Hyv¨arinen et al. (2008). Independent Component Analysis (ICA). 
# 2) Alan J. Izenman (2008). Modern Multivariate Statistical Techniques (MMST).
# 3) David. C. Lay (2011). Linear Algebra and Its Applications. 
FastICA <-  function (X, m, alpha = 1, max_iteration = 200, tol_level = 1e-04) {
  # Implement the fastICA with Parallel algorithm.
  # Args:
  #   X: a data matrix with n rows of observations and r columns of variables. 
  #   m: an integer, specifying the number of componenets to be extracted. 
  #   alpha: a numeric scalar between 1 and 2, evoked only if "logcosh" used in fun. 
  #   max_iteration: an integer, specifying the maximum number of iterations to perform. 
  #   tol_level: a positive scalar, 
  #              specifying the threshold for the convergence of the unmixing matrix W.
  # Return:
  #   A list of matrices:
  #     1) X: the preprocessed data matrix supplied;
  #     2) V: the matrix for whitening;
  #     3) W: the unmixing matrix;
  #     4) A: the mixing matrix. 
  #     5) S: the matrix of original sources separated from X. 
  Whitening <- function(X) {
    # Whiten a matrix.
    # Args:
    #   X: a data matrix, with r (variables) rows and n (observations) columns. 
    # Return:
    #   A list of two: the post-whiten data matrix and the whiten matrix. 
    
    # Obtain the sample covariance matrix S. See the definition of S on Page 42 of LAIA. 
    S <- X %*% t(X) / (n - 1)
    
    # Conduct the singular value decomposition (SVD) on the sample covariance matrix S. 
    # For using SVD to carry out whitening,
    # see the the back of P 556 of Modern Multivariate Statistical Techniques (MMST).
    svd <- La.svd(S)
    eigen_values <- svd$d
    # Obtain the matrix U in (15.1) of MMST. 
    U <- svd$u
    
    # Find the square root inverse matrix of \Gamma^{- 1 / 2} in (15.1) of MMST.
    D <- diag(c(1 / sqrt(eigen_values)))
    
    # Compute the prewhitening matrix that whitens the data matrix. 
    # Note that U is a matrix whose columns contain the left singular vectors of S, 
    # which are also the eigenvectors of S. 
    Whiten_mat <- D %*% t(U)
    
    # Retain the first m principal components of the covariance matrix S, 
    # if dimension reduction is required (n.comp < r).   
    Whiten_mat <- matrix(Whiten_mat[1 : m, ], m, r)
    
    # Perform the whitening on X,
    # where both the pre and post whitened matrix X should be m (variables) by n (observations).
    # See P 557 of MMST for reference. 
    X_whiten <- Whiten_mat %*% X
    
    return(list(post_whiten_matrix = X_whiten, whiten_matrix = Whiten_mat))
  }
  
  Decorrelation <- function(W) {
    # Implement an iterative MDUM decorrelation that avoids matrix inversion.
    # Args:
    #   W: a matrix, on which a symmetrix orthogonalization should be carried out. 
    # Return:
    #   A matrix that is orthogonal. 
    # Specifying the initial value and the convergence criteria. 
    deviation = 1
    threshold = 1e-05
    
    # Normalize W by the maximum absolute column sum norm (L1-norm). 
    W <- W / norm(W)
    while (deviation > threshold) {
      W <- 1.5 * W - 0.5 * W %*% t(W) %*% W
      # Check whether WW' is close to the identity matrix I. 
      deviation <- max(abs(abs(diag(W %*% t(W))) - 1))
    }
    return(W)
  }
  
  ICAParallel <- function (X) {
    # Implement the fastICA with Parallel algorithm.
    # Args:
    #   X: a data matrix with n rows of observations and r columns of variables. 
    # Return:
    #   The unmixing matrix W. 
    
    # Initialzation. 
    X <- t(X)
    n <- nrow(X)
    W <- W_init
    
    # Parallel decorrelation. 
    W <- Decorrelation(W)
    iteration <- 1
    deviation <- 1/ tol_level
    
    while (deviation > tol_level && iteration < max_iteration) {
      # For k = 1, ..., n.comp,
      # Compute w'X. 
      WX <- W %*% t(X)
      
      # Compute XE{g(w'X)}.
      gWX <- tanh(alpha * WX)
      v1 <- gWX %*% X / n
      
      # Compute w'E{g(w'X)}.
      gWX_prime <- alpha * (1 - (gWX) ^ 2)
      v2 <- diag(apply(gWX_prime, 1, FUN = mean)) %*% W
      
      # Update W. 
      W <- v1 - v2
      W <- Decorrelation(W)
      
      # Compute its devatiation from the orthogonal matrix. 
      deviation <- max(abs(abs(diag(W %*% t(W))) - 1))
      message("Iteration ", iteration, ", Deviation = ", format(deviation))
      iteration <- iteration + 1
    }
    # By the description of the fastICA algo on Page570 of MMST, 
    # each column of matrix W represents an unmixing m-dimensional row vector w_k for k = 1, ..., m.
    # See the discussion at the back of P 569 of MMST. 
    
    return(W)
  }
  
  n <- nrow(X)
  r <- ncol(X)

  #################################
  # Error handling.
  if (alpha < 1 || alpha > 2) {
    stop("alpha must be in range [1,2]")
  }

  #################################
  # Determine the numer of componenets. 
  if (m > min(n, r)) {
    message("'n.comp' is too large: reset to ", min(n, p))
    m <- min(n, r)
  }
  
  #################################
  # Initialize the unmixing matrix W as a square matrix with m rows.
  W_init <- matrix(rnorm(m ^ 2), m, m)
  
  ############## Center the data. ################### 
  message("Centering")
  # Function scale centers the columns of a numeric matrix,
  # by subtracting the column means of matrix from the corresponding columns. 
  X <- scale(X)
  
  # Transpose the data matrix s.t. X becomes r by n, as in the standard setting of PCA. 
  # See Page 42 of Linear Algebra and Its Applications (LAIA). 
  X <- t(X)
  
  ############## Whiten the data. ###################
  message("Whitening")
  
  whiten_res <- Whitening(X)
  V <- whiten_res$whiten_matrix
  X_whiten <- whiten_res$post_whiten_matrix
  
  ############## Perform ICA. ###################
  # Apply the Parallel Algorithm of FastICA. 
  W <- ICAParallel(X_whiten)
  
  ############## Compute the outputs. ################### 
  # W is the m by m unmixing matrix, and V is the m by r whiten matrix. 
  # The m by r matrix B is defined as the matrix product of the two. 
  # See Page 210 of ICA (2008). 
  B <- W %*% V
  # S is the m by n source matrix, which can be computed as (15.4) of MMST on Page 560.
  S <- B %*% X
  # A is the r by m mixing matrix, defined in (15.3) of MMST. 
  # The effects of A and B cancle out s.t. BA = I. 
  A <- t(B) %*% solve(B %*% t(B))
  
  # Transpose the output matrices, with n columns correspond to n observations, 
  # to be in line with the definition of the data matrix X. 
  return(list(X = t(X), K = t(V), W = t(W), A = t(A), S = t(S)))
}

############## Data Examples. ###################
################################# Unmixing two mixed independent uniforms.
S <- matrix(runif(10000), 5000, 2)
A <- matrix(c(1, 1, -1, 3), 2, 2, byrow = TRUE)
X <- S %*% A
res <- FastICA(X, m = 2, alpha = 1, max_iteration = 200, tol_level = 0.0001)
par(mfrow = c(1, 3))
plot(res$X, main = "Pre-processed data")
plot(res$X %*% res$K, main = "PCA components")
plot(res$S, main = "ICA components")

#################################  Unmixing two independent signals.
S <- cbind(sin((1 : 1000) / 20), rep((((1 : 200) - 100) / 100), 5))
A <- matrix(c(0.291, 0.6557, -0.5439, 0.5572), 2, 2)
X <- S %*% A
res <- FastICA(X, 2, alpha = 1, max_iteration = 200, tol_level = 0.0001)
par(mfcol = c(2, 3))
plot(1 : 1000, S[, 1], type = "l", main = "Original Signals",
     xlab = "", ylab = "")
plot(1 : 1000, S[, 2], type = "l", xlab = "", ylab = "")
plot(1 : 1000, X[, 1], type = "l", main = "Mixed Signals",
     xlab = "", ylab = "")
plot(1 : 1000, X[, 2], type = "l", xlab = "", ylab = "")
plot(1 : 1000, res$S[, 1], type = "l", main = "ICA source estimates",
     xlab = "", ylab = "")
plot(1 : 1000, res$S[, 2], type = "l", xlab = "", ylab = "")

################################# 
# Using FastICA to perform projection pursuit on a mixture of bivariate normal distributions.
if(require(MASS)){
  x <- mvrnorm(n = 1000, mu = c(0, 0), Sigma = matrix(c(10, 3, 3, 1), 2, 2))
  x1 <- mvrnorm(n = 1000, mu = c(-1, 2), Sigma = matrix(c(10, 3, 3, 1), 2, 2))
  X <- rbind(x, x1)
  res <- FastICA(X, 2, alpha = 1, max_iteration = 200, tol_level = 0.0001)
  par(mfrow = c(1, 3))
  plot(res$X, main = "Pre-processed data")
  plot(res$X %*% res$K, main = "PCA components")
  plot(res$S, main = "ICA components")
}

