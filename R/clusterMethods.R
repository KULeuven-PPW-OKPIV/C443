#' Plot a clusterforest object
#'
#' A function that can be used to plot a clusterforest object, either by returning plots
#' with information on the cluster solutions (e.g., average silhouette width), or plots of the medoid trees of each solution.
#'
#' This function can be used to plot a clusterforest object in two ways. If it's used without specifying a solution,
#' then the average silhouette width, and within cluster similarity measures are plotted for each solution. 
#' If additionally, predictive_plots=TRUE, two more plots are returned, namely a plot showing for each solution the 
#' predictive accuracy when making predictions based on the medoid trees, and a plot showing for each solution the agreement between
#' the class label for each object predicted on the basis of the random forest as a whole versus based on the medoid trees.
#' These plots may be helpful in deciding how many clusters are needed to summarize the forest (see Sies & Van Mechelen, 2020).
#'
#' If the function is used with the clusterforest object and the number of the solution, then the medoid tree(s)
#' of that solution are plotted. 
#' 
#' @param x A clusterforest object
#' @param solution The solution to plot the medoid trees from. Default = NULL
#' @param predictive_plots Indicating whether predictive plots should be returned: A plot showing the predictive accuracy
#' when making predictions based on the medoid trees, and a plot of the agreement between the class label
#' for each object predicted on the basis of the random forest as a whole versus based on the medoid trees. Default = FALSE.
#' @export
#' @importFrom cluster pam
#' @importFrom graphics axis plot mtext
#' @import MASS
#' @references \cite{Sies, A. & Van Mechelen I. (2020). C443: An R-package to see a forest for the trees. Journal of Classification.}
#' @examples
#' require(MASS)
#' require(rpart)
#'#Function to draw a bootstrap sample from a dataset
#'DrawBoots <- function(dataset, i){
#'set.seed(2394 + i)
#'Boot <- dataset[sample(1:nrow(dataset), size = nrow(dataset), replace = TRUE),]
#'return(Boot)
#'}
#'
#'#Function to grow a tree using rpart on a dataset
#'GrowTree <- function(x,y,BootsSample, minsplit = 40, minbucket = 20, maxdepth =3){
#'  controlrpart <- rpart.control(minsplit = minsplit, minbucket = minbucket,
#'  maxdepth = maxdepth, maxsurrogate = 0, maxcompete = 0)
#'  tree <- rpart(as.formula(paste(noquote(paste(y, "~")),
#'  noquote(paste(x, collapse="+")))), data = BootsSample,
#'  control = controlrpart)
#'  return(tree)
#'}
#'
#'#Use functions to draw 20 boostrapsamples and grow a tree on each sample
#'Boots<- lapply(1:10, function(k) DrawBoots(Pima.tr ,k))
#'Trees <- lapply(1:10, function (i) GrowTree(x=c("npreg", "glu",  "bp",
#'  "skin",  "bmi", "ped", "age"), y="type",
#'Boots[[i]] ))
#'
#'ClusterForest<- clusterforest(observeddata=Pima.tr,treedata=Boots,trees=Trees,m=1,
#' fromclus=1, toclus=5, sameobs=FALSE)
#'plot(ClusterForest)
#'plot(ClusterForest,2)
plot.clusterforest <- function(x, ..., solution=NULL, predictive_plots=FALSE) {
  clusters=x$clusters
  medoids=x$medoids
  mds=x$medoidtrees
  sil=x$avgsilwidth
  sums=x$withinsim
  agreement=x$agreement
  accuracy=x$accuracy

  if(is.null(solution)){

    # Within plot
    sums[unlist(lapply(sums , is.null))] <- NA
    M<- unlist(sums)
    withinplot <- plot(M, main="Within-cluster similarity plot", xlab="Number of clusters", ylab="Within-cluster similarity", xlim=c(1,length(medoids)), xaxt="n")
    withinplot<-withinplot + axis(1, at = seq(from = 1, to = length(medoids), by = 1))

    #### Silhouete plot
    sil[unlist(lapply(sil , is.null))] <- NA
    sil<- unlist(sil)
    silplot <- plot(sil, main = "Silhouette plot", xlab = "Number of clusters", ylab = "Average Silhouette width", xlim=c(1,length(medoids)), xaxt="n")
    silplot <- silplot + axis(1, at = seq(from = 1, to = length(medoids), by = 1))


  if(predictive_plots==TRUE){
  ## accuracy
    accuracy[unlist(lapply(accuracy , is.null))] <- NA
    accuracy<- unlist(accuracy)
    accuracyplot <- plot(accuracy, main= "Accuracy of predictions for each solution", xlab = "Number of clusters", ylab = "accuracy", xlim = c(1,length(medoids)), xaxt = "n", ylim=c(0.3,1) )
    accuracyplot<- accuracyplot + axis(1, at = seq(from = 1, to = length(medoids), by = 1)) + mtext(paste("Accuracy full forest = ", accuracy[length(accuracy)-1], ", proportion most frequent class = ", accuracy[length(accuracy)]))
    

    agreement[unlist(lapply(agreement , is.null))] <- NA
    agreement<- unlist(agreement)
    agreementplot <- plot(agreement, main= "Agreement in predicted labels by medoids vs forest ", xlab = "Number of clusters", ylab = "Agreement", xlim = c(1,length(medoids)), xaxt = "n", ylim=c(0.3,1) )
    agreementplot<- agreementplot + axis(1, at = seq(from = 1, to = length(medoids), by = 1))
  
    }
  } else{
    for(i in 1:solution){
      plot(x$medoidtrees[[solution]][[i]])
    }
  }
}
  
  
#' Print a clusterforest object
#'
#' A function that can be used to print a clusterforest object.
#'
#' @param x A clusterforest object
#' @param solution The solution to print the medoid trees from. Default = NULL
#' @param ... Additional arguments
#' @export
print.clusterforest<- function(x, ..., solution=1){
  print(unlist(x$medoidtrees[solution], recursive=FALSE))
  cat("Cluster to which each tree is assigned: " ,unlist(x$clusters[solution], recursive=FALSE))
}

#' Summarize a clusterforest object
#'
#' A function to summarize a clusterforest object.
#' @param object A clusterforest object
#' @param ... Additional arguments
#' @export
summary.clusterforest<- function(object, ...){
  cat("Solutions checked: " , length(object$medoids), "\n")
  cat("Average Silhouette Width per solution: \n" , unlist(object$avgsilwidth),  "\n")
  cat("Within cluster similarity per solution:\n " , unlist(object$withinsim),  "\n")
  cat("Agreement predicted labels medoids vs forest per solution:\n " , unlist(object$agreement),  "\n")
}

#' Get the cluster assignments for a solution of a clusterforest object
#'
#' A function to get the cluster assignments for a given solution of a clusterforest object.
#' @param clusterforest A clusterforest object
#' @param solution The solution for which cluster assignments should be returned. Default = 1
#' @export
clusters <- function(clusterforest, solution){
  UseMethod("clusters",clusterforest)
}


#' Get the cluster assignments for a solution of a clusterforest object
#'
#' A function to get the cluster assignments for a given solution of a clusterforest object.
#' @param clusterforest The clusterforest object
#' @param solution The solution
#' @export
clusters.default <- function(clusterforest, solution)
{
  print("Make sure that the clusterforest argument is an object from class clusterforest.")
}

#' Get the cluster assignments for a solution of a clusterforest object
#'
#' A function to get the cluster assignments for a given solution of a clusterforest object.
#' @param clusterforest The clusterforest object
#' @param solution The solution
#' @export
clusters.clusterforest<- function(clusterforest, solution=1){
  return(unlist(clusterforest$clusters[solution], recursive=FALSE))
}

#' Get the medoid trees for a solution of a clusterforest object
#'
#' A function to get the medoid trees for a given solution of a clusterforest object.
#' @param clusterforest A clusterforest object
#' @param solution The solution for which medoid trees should be returned. Default = 1
#' @export
medoidtrees <- function(clusterforest, solution){
  UseMethod("medoidtrees",clusterforest)
}


#' Get the medoid trees for a solution of a clusterforest object
#'
#' A function to get the medoid trees for a given solution of a clusterforest object.
#'
#' @param clusterforest A clusterforest object
#' @param solution The solution for which medoid trees should be returned. Default = 1
#' @export
medoidtrees.default <- function(clusterforest, solution)
{
  print("Make sure that the clusterforest argument is an object from class clusterforest.")
}


#' Get the medoid trees for a solution of a clusterforest object
#'
#' A function to get the medoid trees for a given solution of a clusterforest object.
#'
#' @param clusterforest A clusterforest object
#' @param solution The solution for which medoid trees should be returned. Default = 1
#' @export
medoidtrees.clusterforest<- function(clusterforest, solution=1){
  return(unlist(clusterforest$medoidtrees[solution], recursive=FALSE))
}

#' Get the similarity matrix that wast used to create a clusterforest object
#'
#' A function to get the similarity matrix used to obtain a clusterforest object.
#'
#' @param clusterforest A clusterforest object
#' @export
treesimilarities <- function(clusterforest){
  UseMethod("medoidtrees",clusterforest)
}

#' Get the similarity matrix that wast used to create a clusterforest object
#'
#' A function to get the similarity matrix used to obtain a clusterforest object.
#'
#' @param clusterforest A clusterforest object
#' @export
treesimilarities.default <- function(clusterforest)
{
  print("Make sure that the clusterforest argument is an object from class clusterforest.")
}

#' Get the similarity matrix that wast used to create a clusterforest object
#'
#' A function to get the similarity matrix used to obtain a clusterforest object.
#'
#' @param clusterforest A clusterforest object
#' @export
treesimilarities.clusterforest<- function(clusterforest){
  return(clusterforest$treesimilarities)
}
