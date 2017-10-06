# Agglomerative Clustering using single, complete or average linkage. 
# Author: Michael Ruochen Zeng, micalzrc@gmail.com
rm(list = ls())

###################################################### Implement functions ################################################
Euc_dist <- function(x, y) {
  # Compute the Euclidean distance between two observations. 
  #
  # Args:
  #   x: a vector, representing an obs. 
  #   y: a vector, representing an obs. 
  # Returns:
  #   A scalar, indicating the Euclidean distance between two obs.  
  sqrt(sum(x - y ) ^ 2)
}

Linkage <- function(c1, c2, fun = min) {
  # Compute the distance between two clusters. 
  #
  # Args:
  #   c1: a matrix, consisting of the obs. of the 1st cluster. 
  #   c2: a matrix, consisting of the obs. of the 2nd cluster. 
  #   fun: a function, specifying the way to compute the distance between two clusters,
  #        could be min, max, mean,
  #        corresponding to single linkage, complete linkage and average linkage. 
  # Returns:
  #   A vector of character strings.  
  Distance_cluster <- function() {
    # Compute the elementwise distance between two clusters. 
    #
    # Returns:
    #   A matrix, 
    #   whose (i, j)th element is the distance between the ith obs. in Cluster 1 and jth obs. in Cluster 2. 
    dists <- lapply(1 : dim(c1)[1], 
                    function(j) sapply(1 : dim(c2)[1], 
                                       function(i) Euc_dist(c2[i, ], c1[j, ])))
    return(Reduce(rbind, dists))
  }
  return(fun(Distance_cluster()))
}

Keep_name <- function(index, list) {
  # Return the names of elements in a list when rbind is applied. 
  #
  # Args:
  #   index: a vector, indicating which part of the list is extracted to perform rbind. 
  #   list: the list of interest. 
  # Returns:
  #   A vector of character strings.  
  Extract_name <- function(i) {
    if (is.matrix(list[[i]])) {
      n <- rownames(list[[i]])
    } else {
      n <- names(list)[i]
    }
    return(n)
  }

  names <- lapply(index, Extract_name)
  return(Reduce(c, names))
}

######################################################## Implement the Algo ###############################################
Agglo_cluster <- function(fun) {
  force(fun)
  
  function(depth, dat = dataset) {
    # Conduct Agglomerative Clustering according to fun. 
    #
    # Args:
    #   depth: an integer, indicating the depth of the algo from bottom up. 
    #   dat: a data frame, consisting of the data matrix of interest. 
    # Returns:
    #   A list, consisting of clusters formed.  
    Switch2Mat <- function(x) {
      # Switch an atomic vector to a matrix. 
      #
      # Args:
      #   x: either a matrix or a atomic vector.  
      # Returns:
      #   A matrix. 
      if (!is.matrix(x)) {
        x <- t(as.matrix(x))
      }
      return(x)
    }
    cls <- as.list(as.data.frame(t(dat)))
    
    for (i in 1 : depth) {
      # Compute the distance matrix. 
      num_cls <- length(cls)
      dist_list <- lapply(1 : num_cls, 
                          function(i) sapply(1 : num_cls, 
                                             function(j) Linkage(Switch2Mat(cls[[i]]), 
                                                                 Switch2Mat(cls[[j]]), fun)))
      dist_mat <- Reduce(rbind, dist_list)
      diag(dist_mat) <- max(dist_mat) + 1
      
      # Find the pair of obs. with largest similarity. 
      max_index <- which(dist_mat == min(dist_mat), arr.ind = T)
      new_index <- unname(max_index[1, ])
      
      # Form a new cluster using this pair of obs. 
      cl_new <- cls[new_index]
      cl_mat <- Reduce(rbind, cl_new)
      rownames(cl_mat) <- Keep_name(index = new_index, list = cls)
      
      # Update the list of clusters. 
      cls <- append(cls[- new_index], list(cl_mat))
    }
    return(cls)
  }
}

######################################################## Data Examples ####################################################
dataset = mtcars
cl_res <- Agglo_cluster(mean)(31)
