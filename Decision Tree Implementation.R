# Classification Decision Tree Model Using Continuous and Ordinal Predicting Variables.
# Author: Michael Ruochen Zeng, micalzrc@gmail.com
library(data.table)
library(mlbench)
library(stringr)
library(gmodels)
rm(list = ls())

####################################### Implement Functions ####################################################
# Calculate the Gini index for a split dataset. 
Gini_index <- function(groups, class_values) {
  # Calculate the Gini index for a partition of groups.
  #
  # Args:
  #   groups: a list of two data tables for binary classification case. 
  #   class_values: a vector containing unique values of the class values for classification. 
  # Returns:
  #   A numeric scalar, indicating the Gini index for the partition of the groups. 
  
  Group_Gini <- function(class_value) {
    # Create a closure for a specific class value. 
    #
    # Args:
    #   class_value: a numeric value, indicating the class. 
    # Returns:
    #   A closure with arguement group. 
    force(class_value)
    
    function(group) {
      # Compute the Gini index for a speific group for a specific class value.  
      #
      # Args:
      #   group: a data table, consisting of the observations for one group. 
      # Returns:
      #   A numeric scalar, 
      #   indicating the Gini index for the specific combination of group and class. 
      size <- nrow(group)
      if (size == 0) {
        gini <- 0
      } else {
        labels <- group[[ncol(group)]]
        p <- sum(labels == class_value) / size
        gini <- p * (1 - p)
      }
      return(gini)
    }
  }
  classes <- lapply(class_values, Group_Gini)
  ginis <- sum(unlist(lapply(classes, function(fun) lapply(groups, fun))))
  return(ginis)
}

# Test Gini_index() function. 
#dt1 <- data.table(c(1, 1), c(1, 0))
#dt2 <- data.table(c(1, 1), c(1, 0))
#dt1 <- data.table(c(1, 1), c(1, 1))
#dt2 <- data.table(c(1, 1), c(0, 0))

#groups <- list(dt1, dt2)
#class_values <- c(0, 1)
#Gini_index(groups, c(0, 1))

# Split a dataset based on an attribute and an attribute value.
test_split <- function(index, split_value, dataset) {
  # Partition the dataset into two parts.
  #
  # Args:
  #   index: an integer, specifying the specific column / variable to look at. 
  #   split_value: a numeric scalar, specifying the split value for the column,
  #          according to which the dataset is splited. 
  #   dataset: a data table to be splited. 
  # Returns:
  #   A list of two data tables. 
  pos <- dataset[[index]] < split_value
  left <- dataset[pos]
  right <- dataset[!pos]
  return (list(left = left, right = right))
}

# Select the best split point for a dataset
Get_split <- function(dataset) {
  # Get the best split for the dataset.
  #
  # Args:
  #   dataset: a data table to be splited. 
  # Returns:
  #   A list containin the information of the optimal split. 
  Split <- function(index) {
    # Partition the dataset into two parts according to the values of a variable specified by index.
    #
    # Args:
    #   index: an integer, specifying the specific column / variable to look at. 
    # Returns:
    #   A closure with argument i. 
    force(index)
    
    function(i) {
      # Partition the dataset into two parts according to the specific value of a variable.
      #
      # Args:
      #   i: an integer, specifying the specific row of variable to look at. 
      # Returns:
      #   A list of four, 
      #   correpsonding to which variable, which value to split, the optimal Gini index value,
      #   and the list of two data tables after split. 
      force(i)
      
      split_value <- dataset[[index]][i]
      groups_dt <- test_split(index, split_value, dataset)
      gini <- Gini_index(groups_dt, class_values)
      return(list(index = index, value = split_value, groups = groups_dt, score = gini))
    }
  }
  # Subset the last column of dataset for the target variable. 
  # col_index <- names(dataset)
  class_values <- sort(unique(dataset[[ncol(dataset)]]))
  
  # Get all possible splits. 
  vars_funs <- lapply(1 : (ncol(dataset) - 1), Split)
  splits <- unlist(lapply(vars_funs, function(fun) lapply(1 : nrow(dataset), fun)), recursive = F)
  lapply(splits, function(x) print(paste0("X", x$index, " < ", x$value, " and Gini index =", x$score)))
  
  # Get the best split based on the smallest Gini index. 
  pos <- which.min(sapply(splits, function(x) x$score))
  best_split <- splits[[pos]]
  return(best_split)
}

# Create a terminal node value.
Mode <- function(x) {
  # Find the mode of elements in a vector.
  #
  # Args:
  #   x: a numeric vector, whose mode is to be found. 
  # Returns:
  #   A numeric scalar specifying the mode of x. 
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

Get_terminal <- function(group) {
  # Specify a terminal node for a tree model. 
  #
  # Args:
  #   group: a data table, representing the data contained in a node. 
  # Returns:
  #   A list with three elements, added to a normal child node:
  #   1) class: the final class prediction of the terminal node;
  #   2) class_count: the count of the observations belonging to the class predicted;
  #   3) total_count: the total count of the observations in this terminal node. 
  col_names <- names(group)
  outcomes <- group[[col_names[ncol(group)]]]
  terminal <- list(class = Mode(outcomes), 
                   class_count = sum(outcomes == Mode(outcomes)), 
                   total_count = nrow(group))
  return(terminal)
}

# Create child splits for a node or make terminal.
Get_child_split <- function(node, max_depth, min_size, depth = 1) {
  # Obtain child nodes from a mother node.
  #
  # Args:
  #   node: a list representing the node, which is to be splited.
  #   max_depth: an integer, 
  #              specifying the maximum number splits to be made from the node. 
  #   min_size: an integer, 
  #             specifying the minimial size of a terminal node.
  #   depth: an integer, used as indicator variable, 
  #          indicating the current depth of the splits from the node. 
  # Returns:
  #   A numeric scalar specifying the mode of x. 
  left <- node[[3]][[1]]
  right <- node[[3]][[2]]
  
  # Check for a no split.
  if (nrow(left) == 0 || nrow(right) == 0) {
    node[['left']] <- Get_terminal(rbind(left, right))
    node[['right']] <- Get_terminal(rbind(left, right))
    return(node)
  }
  
  # Check for max depth.
  if (depth >= max_depth) {
    node[['left']] <- Get_terminal(left)
    node[['right']] <- Get_terminal(right)
    return(node)
  }
  
  # Process left child.
  if (nrow(left) <= min_size) {
    node[['left']] <- Get_terminal(left)
  } else {
    node[['left']] <- Get_split(left)
    node[['left']] <- Get_child_split(node[['left']], max_depth, min_size, depth + 1)
  }

  # Process right child.
  if (nrow(right) <= min_size) {
    node[['right']] <- Get_terminal(right)
  } else {
    node[['right']] <- Get_split(right)
    node[['right']] <- Get_child_split(node[['right']], max_depth, min_size, depth + 1)
  }
  return(node)
}

# Build a decision tree.
Build_tree <- function(dataset, max_depth, min_size) {
  # Build a decision model.
  #
  # Args:
  #   dataset: a data table, consisting of the dataset for building the model.
  #   max_depth: an integer, 
  #              specifying the maximum number splits to be made from the node. 
  #   min_size: an integer, 
  #             specifying the minimial size of a terminal node.
  # Returns:
  #   An object of lists of list, with each list specifying a split. 
  root <- Get_split(dataset)
  tree <- Get_child_split(root, max_depth, min_size)
  return(tree)
}

# Print a decision tree.
Print_tree <- function(node, model = tree) {
  # Print out a decision model.
  #
  # Args:
  #   node: a list representing a node. 
  #   model: a list representing a decision tree model. 
  # Returns:
  #   A list printed out, representing the tree structure of model. 
  if (length(node) == length(model)) {
    print(paste0('X', node[['index']], " < ", node[['value']]))
    Print_tree(node[['left']])
    Print_tree(node[['right']])
  } else {
    print(node$class)
  }
}

# Make a prediction with a decision tree.
Predict_tree <- function(data, node, model = tree) {
  # Predict the outcome given one instance of observation. 
  #
  # Args:
  #   node: a list representing a node. 
  #   data: a vector representing an observation. 
  #   model: a list representing a decision tree model. 
  # Returns:
  #   The predicted class for the input observation. 
  # Check if the node goes left, as specified by the if condition.
  if (data[node[['index']]] < node[['value']]) {
    # Check if the left child node is a internal node. 
    if (length(node[['left']]) == length(model)) {
      Predict_tree(data, node[['left']], model)
    } else {
      # If the node is terminal node, print out the result directly.   
      node[['left']]$class
    }
  # Else we goes right. 
  } else {
    # Check if the right child node is a internal node. 
    if (length(node[['right']]) == length(model)) {  
      Predict_tree(data, node[['right']], model)
    } else {
      # If the node is terminal node, print out the result directly.     
      node[['right']]$class
    }
  }
}  

# Evaluate the out-of-sample performance of a decision model using testing dataset. 
Evaluate_tree <- function(tree, data) {
  # Evaluate the out-of-sample performance of a decision tree using testing sample. 
  #
  # Args:
  #   tree: a list representing a decision tree model. 
  #   data: a vector representing an observation. 
  # Returns:
  #   The predicted class for the input observation.
  pred <- unlist(lapply(1 : nrow(data), 
                        function(i) Predict_tree(unlist(unclass(data[i])), 
                                                 node = tree, model = tree)))
  for (i in 1 : nrow(data)) {
    print(paste0('For observation', i, ", the predicted class is ", 
                 pred[i], " and the true class is ", data[[ncol(data)]][i]))
  }
  return(pred)
}

####################################### Tony Example ###########################################################
dataset = c(2.771244718,1.784783929,0,
            1.728571309,1.169761413,0,
            3.678319846,2.81281357,0,
            3.961043357,2.61995032,0,
            2.999208922,2.209014212,0,
            7.497545867,3.162953546,1,
            9.00220326,3.339047188,1,
            7.444542326,0.476683375,1,
            10.12493903,3.234550982,1,
            6.642287351,3.319983761,1)
dataset <- t(matrix(dataset, nrow = 3))
dataset <- as.data.table(dataset)
split <- Get_split(dataset)

tree <- Build_tree(dataset, max_depth = 3, min_size = 1)
Print_tree(tree)
pred <- Evaluate_tree(tree, dataset)
CrossTable(dataset$V3, pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('True Class', 'Predicted Class'))

####################################### Real Example ###########################################################
####################################### Configuration Data #####################################################
# Banknote authentication.
dat_banknote <- fread("C:/Users/mzeng/Desktop/data_banknote_authentication.csv", header = F, stringsAsFactors = T)
dat <- dat_banknote
y_index = 5
mp <- 5
ms <- 10
dat[[y_index]] <- dat[[y_index]] + 1

# Sonor: classification of metal and rock. 
dat_sonar <- fread("C:/Users/mzeng/Desktop/sonar.all-data.csv", header = F, stringsAsFactors = T)
dat <- dat_sonar
y_index = 61
mp <- 6
ms <- 2

# Breast cancer data: classification of benign and malignant. 
data(BreastCancer)
dat_BreastCancer <- BreastCancer[, -1]
dat <- as.data.table(dat_BreastCancer)
y_index = 10
mp <- 6
ms <- 2

# Iris data:  
data(iris)
dat_iris <- iris[1 : 100, ]
dat <- as.data.table(dat_iris)
y_index = 5
mp <- 6
ms <- 2

####################################### Data Preprocessing #####################################################
# Generic code.
# set.seed(10)
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
# Note: the target variable should be included as the last column of the dataset dat. 

# Data partition. 
train_dat <- dat[samp, ]
test_dat <- dat[- samp, ]
if (nrow(train_dat) > nrow(test_dat)) {
  train_dat <- train_dat[- 1 ,]
} else if (nrow(train_dat) < nrow(test_dat)) {
  test_dat <- test_dat[- 1 ,]
}

####################################### Model Training and Testing #############################################
# Build a decision tree using the training sample.  
tree <- Build_tree(train_dat, max_depth = mp, min_size = ms)
Print_tree(tree)

# Evaluate the out-of-sample performance using the testing sample. 
pred <- Evaluate_tree(tree, test_dat)
t <- CrossTable(test_dat$y, pred,
                prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
                dnn = c('True Class', 'Predicted Class'))
(accuracy <- 1 - (t$t[1, 2] + t$t[2, 1]) / sum(t$t))
