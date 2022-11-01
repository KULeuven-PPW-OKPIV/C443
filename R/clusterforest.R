#' Clustering the classification trees in a forest based on similarities
#'
#' A function to get insight into a forest of classification trees by clustering the trees in a forest using Partitioning Around Medoids (PAM, Kaufman & Rousseeuw, 2009), based on user provided similarities,
#' or based on similarities calculated by the package using a similarity measure chosen by the user (see Sies & Van Mechelen, 2020).
#'
#' The user should provide the number of clusters that the solution should contain, or a range of numbers that should be explored.
#' In the latter case, the resulting clusterforest object will contain clustering results for each solution.
#' On this clusterforest object, several methods, such as plot, print and summary, can be used.
#'
#' @param observeddata The entire observed dataset
#' @param treedata A list of dataframes on which the trees are based. Not necessary if the data set is included in the tree object.
#' @param trees A list of trees of class party, classes inheriting from party (e.g., glmtree), classes that can be coerced to party (i.e., rpart, Weka_tree, XMLnode), or a randomForest or ranger object.
#' @param simmatrix A similaritymatrix with the similarities between all trees. Should be square, symmetric and have ones on the diagonal. Default=NULL
#' @param m Similarity measure that should be used to calculate similarities, in the case that no similarity matrix was provided by the user. Default=NULL.
#' m=1 is based on counting common predictors;
#' m=2 is based on counting common predictor-split point combinations;
#' m=3 is based on common ordered sets of predictor-range part combinations (see Shannon & Banks (1999));
#' m=4 is based on the agreement of partitions implied by leaf membership (Chipman, 1998);
#' m=5 is based on the agreement of partitions implied by class labels (Chipman, 1998);
#' m=6 is based on the number of predictor occurrences in definitions of leaves with same class label;
#' m=7 is based on the number of predictor-split point combinations in definitions of leaves with same class label
#' m=8  measures closeness to logical equivalence (applicable in case of binary predictors only)
#' @param tol A vector with for each predictor a number that defines the tolerance zone within which two split points of the predictor in question are assumed equal. For example, if the tolerance for predictor X
#' is 1, then a split on that predictor in tree A will be assumed equal to a split in tree B as long as the splitpoint in tree B is within the splitpoint in tree A + or - 1. Only applicable for m=1 and m=6. Default=NULL
#' @param weight If 1, the number of dissimilar paths in the Shannon and Banks measure (m=2), should be weighted by 1/their length (Otherwise they are weighted equally). Only applicable for m=2. Default=NULL
#' @param fromclus The lowest number of clusters for which the PAM algorithm should be run. Default=1.
#' @param toclus The highest number of clusters for which the PAM algorithm should be run. Default=1.
#' @param treecov A vector/dataframe with the covariate value(s) for each tree in the forest (1 column per covariate).
#' @param sameobs Are the same observations included in every tree data set? For example, in the case of subsamples or bootstrap samples, the answer is no. Default=FALSE
#' @param seed A seed number that should be used for the multi start procedure (based on which initial medoids are assigned). Default=NULL.
#' @return The function returns an object of class clusterforest, with attributes:
#' \item{medoids}{the position of the medoid trees in the forest (i.e., which element of the list of partytrees)}
#' \item{medoidtrees}{the medoid trees}
#' \item{clusters}{The cluster to which each tree in the forest is assigned}
#' \item{avgsilwidth}{The average silhouette width for each solution (see Kaufman and Rousseeuw, 2009)}
#' \item{accuracy}{For each solution, the accuracy of the predicted class labels based on the medoids.}
#' \item{agreement}{For each solution, the agreement between the predicted class label for each observation based on the forest as a whole, and those based on the
#' medoids only (see Sies & Van Mechelen,2020)}
#' \item{withinsim}{Within cluster similarity for each solution (see Sies & Van Mechelen, 2020)}
#' \item{treesimilarities}{Similarity matrix on which clustering was based}
#' \item{treecov}{covariate value(s) for each tree in the forest}
#' \item{seed}{seed number that was used for the multi start procedure (based on which initial medoids were assigned)}
#' @export
#' @importFrom cluster pam
#' @importFrom partykit nodeids data_party id_node kids_node varid_split split_node index_split breaks_split right_split node_party
#' @importFrom stats predict
#' @importFrom graphics axis plot
#' @importFrom parallel detectCores makeCluster clusterExport parSapply stopCluster
#' @importFrom igraph graph_from_incidence_matrix max_bipartite_match
#' @import MASS
#' @import partykit
#' @import rpart
#'
#' @references \cite{Kaufman, L., & Rousseeuw, P. J. (2009). Finding groups in data: an introduction to cluster analysis (Vol. 344). John Wiley & Sons.}
#' @references \cite{Sies, A. & Van Mechelen I. (2020). C443: An R-package to see a forest for the trees. Journal of Classification.}
#' @references \cite{Shannon, W. D., & Banks, D. (1999). Combining classification trees using MLE. Statistics in medicine, 18(6), 727-740.}
#' @references \cite{Chipman, H. A., George, E. I., & McCulloh, R. E. (1998). Making sense of a forest of trees. Computing Science and Statistics, 84-92.}
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
#'  controlrpart <- rpart.control(minsplit = minsplit, minbucket = minbucket, maxdepth = maxdepth,
#'   maxsurrogate = 0, maxcompete = 0)
#'  tree <- rpart(as.formula(paste(noquote(paste(y, "~")), noquote(paste(x, collapse="+")))),
#'  data = BootsSample, control = controlrpart)
#'  return(tree)
#'}
#'
#'#Use functions to draw 20 boostrapsamples and grow a tree on each sample
#'Boots<- lapply(1:10, function(k) DrawBoots(Pima.tr ,k))
#'Trees <- lapply(1:10, function (i) GrowTree(x=c("npreg", "glu",  "bp",  "skin",
#' "bmi", "ped", "age"), y="type", Boots[[i]] ))
#'
#'#Clustering the trees in this forest
#'ClusterForest<- clusterforest(observeddata=Pima.tr,treedata=Boots,trees=Trees,m=1,
#'fromclus=1, toclus=5, sameobs=FALSE)


clusterforest <- function (observeddata, treedata=NULL, trees, simmatrix=NULL, m=NULL, tol=NULL, weight=NULL,fromclus=1, toclus=1, treecov=NULL, sameobs=FALSE, seed=NULL){
  ############################## Check forest input #####################
  ###  Some checks whether correct forest information is provided by user
  if(typeof(trees) != "list" ) {
    cat("trees must be a list of party tree objects, a list of trees that can be converted to party objects, or a randomforest or ranger object")
    return(NULL)
  }
  
  if('ranger' %in% class(trees)){
      trees<-sapply(1:trees$num.trees, function (k) ranger2party(observeddata, trees, k))
  }else if('randomForest' %in% class(trees)){
        trees<-sapply(1:trees$ntree, function (k) randomForest2party(observeddata, trees, k))
    }else if(!'party' %in% class(trees[[1]])){
     tryCatch(trees<- lapply(1:length(trees), function (i) as.party(trees[[i]])),
             error=function(e){cat("trees must be a list of party tree objects or objects that can be coerced to party trees")})
    }
  
   
  if(!is.null(trees[[1]]$data)){
    treedt<- lapply(1:length(trees), function(k) trees[[k]]$data)
  }
  
  if(is.null(trees[[1]]$data)){
  
    if(is.null(treedata)){
      cat("please submit treedata")
      return(NULL)
      }
    
    if(typeof(treedata) != "list" || class(treedata[[1]]) != "data.frame") {
    cat("please submit correct treedata: a list of data frames on which the trees were grown")
    return(NULL)
  }

  if(length(treedata) != length(trees)){
    cat("the number of data frames provided must be the same as the number of trees")
    return(NULL)
  }
    treedt=treedata
  }
  
  
  
  ## Turn user provided forest information into object of class forest.
  forest <- list(partytrees = trees, treedata = treedt, observeddata=observeddata)

  class(forest) <- append(class(forest), "forest")

  #########################################################################
  ########################## Calculate Similarities #######################
  # Check whether at least one of the arguments is not null (user provided
  # similarity matrix or similarity measure)
  if (is.null(simmatrix) & is.null(m)){
    cat("Either a similarity matrix (simmatrix) should be provided, or
        a similarity measure (m) should be chosen")
    return(NULL)
  }
  # If user provided similarity matrix, check whether it is a square matrix,
  # whether it's symmetric and whether it contains ones on the diagonal.
  if (!is.null(simmatrix)){
    #check square
    if(nrow(simmatrix) != ncol(simmatrix)){
      cat("The similarity matrix should be a square matrix")
      return(NULL)
    }
    #check symmetric
    if(!isSymmetric(simmatrix)){
      cat("The similarity matrix should be a symmetric")
      return(NULL)
    }
    #check ones on diagonal
    if(!sum(diag(simmatrix) == 1) == nrow(simmatrix)){
      cat("The similarity matrix should have ones on the diagonal")
      return(NULL)
    }

    if(nrow(simmatrix) != length(trees)){
      cat("The similarity matrix should have the same dimensions as the number of trees in the forest")
      return(NULL)
    }

    #turn into treesimilarities object
    treesimilarities <- simmatrix
    attr(treesimilarities, "class") <- "treesimilarities"
  }

  # If user didn't provide similarity matrix -- calculate using chosen measure
  if (is.null(simmatrix) & !is.null(m)){

    X= unlist(unique(lapply(1:length(forest$partytrees), function (k) attr(forest$partytrees[[k]]$terms, "term.labels"))))
    Y= unlist(lapply(1:length(forest$partytrees), function(k) colnames(forest$treedata[[k]])[!colnames(forest$treedata[[k]]) %in% X]))

    if (m == 1){
      sim <- M1(forest$observeddata, forest$treedata, Y, X, forest$partytrees, tol=NULL)
    }

    if(m == 2){
      if(is.null(tol)){
        cat("Please provide a tolerance zone for each predictor")
        return(NULL)
      } else{
      sim <- M1(forest$observeddata, forest$treedata, Y, X, forest$partytrees, tol)
      }
    }

    if (m == 3){
      Xclass<- sapply(1:length(X), function(i) class(forest$observeddata[,X[i]] ) == "integer"|class(forest$observeddata[,X[i]] ) == "numeric")

      if("FALSE" %in% Xclass){
        cat("This measure only works on numerical splitvariables (class integer or numeric)")
        return(NULL)
      } else{
        sim <- M2(forest$observeddata, forest$treedata, Y, forest$partytrees, weight)
      }
    }

    if (m == 4){
      sim <- M4(forest$observeddata, forest$treedata, Y, forest$partytrees,sameobs)
    }

    if (m == 5){
      sim <- M3(forest$observeddata, forest$treedata, Y,forest$partytrees, sameobs)
    }



    if (m == 6){
      Xclass<- sapply(1:length(X), function(i) class(forest$observeddata[,X[i]] ) == "integer"|class(forest$observeddata[,X[i]] ) == "numeric")

      if("FALSE" %in% Xclass){
        cat("This measure only works on numerical splitvariables (class integer or numeric)")
        return(NULL)
      }

      sim <- M6(forest$observeddata, forest$treedata, Y, X, forest$partytrees, tol=NULL)
    }

    if(m==7){
      Xclass<- sapply(1:length(X), function(i) class(forest$observeddata[,X[i]] ) == "integer"|class(forest$observeddata[,X[i]] ) == "numeric")

      if("FALSE" %in% Xclass){
        cat("This measure only works on numerical splitvariables (class integer or numeric)")
        return(NULL)
      }

      if(is.null(tol)){
        cat("Please provide a tolerance zone for each predictor")
        return(NULL)
      } else{
            sim <- M6(forest$observeddata, forest$treedata, Y, X, forest$partytrees, tol)
      }
    }

    if (m == 8){


      Xclass<- sapply(1:length(X), function(i) class(forest$fulldata[,X[i]] ) == "factor")

      if("FALSE" %in% Xclass){
        cat("This measure only works on binary splitvariables (class factor)")
        return(NULL)

      }else{

        Xbin<- sapply(1:length(X), function(i) levels(forest$fulldata[,X[i]]) > 2 )
        if("FALSE" %in% Xbin){
          cat("This measure only works on binary splitvariables (class factor)")
          return(NULL)
        } else{
          sim <- M5(forest$fulldata, forest$treedata, Y, X, forest$partytrees, sameobs)
        }
      }
    }

    # Turn similarity matrix into treesimilarities object
    treesimilarities <- sim

    attr(treesimilarities, "class") <- "treesimilarities"
  }

  ################################# End similarities #######################

  ############################## Clustering ################################
  trees=forest$partytrees
  treedata=forest$treedata
  observeddata=forest$observeddata

  medoids<- list(0)
  clusters<- list(0)
  mds<- list(0)
  sil<- list(0)
  agreement<-list(0)
  sums<- list(0)
  meds<- list(0)
  obj<- c(0)
  medsseeds<- list(0)
  correct<- list(0)

  for (i in fromclus:toclus){

    # first do the clustering with BUILD and SWAP phase
    clustering <- pam(x = 1 - treesimilarities, k = i, diss = TRUE, pamonce=3)

    if(is.null(seed)){
      seed<- round(runif(1,0,100000))
    }
    #Then try 1000 random starts + SWAP Phase
    for (j in 1:1000){
      set.seed(seed+j)
      initmedoids= sample(1:nrow(treesimilarities),i)
      medsseeds[[j]] <- pam(1-treesimilarities,k=i,medoids=initmedoids, diss=TRUE, pamonce=3)
      obj[j]<-medsseeds[[j]]$objective[2]
    }

    # check whether objective function of one of the multistarts is better than the one of the build alogrithm
    # and if so, continue with the best result
     if( round(min(obj), 10) < round(clustering$objective[2], 10) ){
      clustering<- medsseeds[[  which.min(round(min(obj), 10))]]
     }


    medoids[[i]] <- clustering $ medoids
    clusters[[i]] <- clustering $ clustering

    meds<- list(0)
    for(j in 1:i){
      meds[[j]] <- trees[[medoids[[i]][j]]]
    }

    mds[[i]]<- meds
    sil[[i]] <-  clustering $ silinfo $ avg.width

    pamtrees <- lapply(1:length(trees), function (i) pamtree(observeddata,treedata[[i]], Y[i],trees[[i]]))
    g<- lapply(1:length(trees), function(k) pamtrees[[k]]$predresp)
    forestpred <- sapply(1:nrow(observeddata), function (k) levels(g[[1]])[which.max( c(sum(unlist(lapply(g, `[[`, k)) ==  levels(unlist(lapply(g, `[[`, k)))[1], na.rm=TRUE),  sum(unlist(lapply(g, `[[`, k)) ==  levels(unlist(lapply(g, `[[`, k)))[2], na.rm=TRUE) )   )] )

    gmed <- g[c(medoids[[i]])]
    w <- table(clusters[[i]])
    
    
    #unweighted
    #medpred1 <- sapply(1:nrow(observeddata), function (k) levels(gmed[[1]])[which.max( c(sum(unlist(lapply(gmed, `[[`, k)) ==  levels(unlist(lapply(gmed, `[[`, k)))[1], na.rm=TRUE),  sum(unlist(lapply(gmed, `[[`, k)) ==  levels(unlist(lapply(gmed, `[[`, k)))[2], na.rm=TRUE) )   )] )
    
    #weighted
    medpred <-sapply(1:nrow(observeddata), function (k) levels(gmed[[1]])[which.max( c(sum(as.numeric(unlist(lapply(gmed, `[[`, k)) ==  levels(unlist(lapply(gmed, `[[`, k)))[1]) *w , na.rm=TRUE),  sum(as.numeric(unlist(lapply(gmed, `[[`, k)) ==  levels(unlist(lapply(gmed, `[[`, k)))[2]) * w, na.rm=TRUE) )   )] )
    
    
    agreement[[i]] <- mean(forestpred == medpred)
    
    correct[[i]] <- mean(forest$observeddata[,Y][1]== medpred)

    sumw<- numeric(0)
    for (j in 1:i){
      sumw[j] <- sum(treesimilarities[clusters[[i]]==j, medoids[[i]][j]])
    }
    sums[[i]]<- sum(sumw) / dim(treesimilarities)[1]
  }

  #Accuracy of forest as a whole and of marginally best class
  medpred <-sapply(1:nrow(observeddata), function (k) levels(gmed[[1]])[which.max( c(sum(as.numeric(unlist(lapply(gmed, `[[`, k)) ==  levels(unlist(lapply(gmed, `[[`, k)))[1]) *w , na.rm=TRUE),  sum(as.numeric(unlist(lapply(gmed, `[[`, k)) ==  levels(unlist(lapply(gmed, `[[`, k)))[2]) * w, na.rm=TRUE) )   )] )
  
  correct[[i+1]] <- mean(forest$observeddata[,Y][1]== forestpred)
  correct[[i+2]] <- max(table(forest$observeddata[,Y][1]))/sum(table(forest$observeddata[,Y][1]))
  
  value <- list(medoids = medoids, medoidtrees = mds, clusters=clusters, avgsilwidth=sil, agreement=agreement, accuracy=correct, withinsim=sums, treesimilarities=treesimilarities, treecov=treecov, seed=seed)
  attr(value, "class") <- "clusterforest"
  return(value)

}



################# subfunctions #############################
#Measure 1: number of splitvariables & splitpoints in common/total number of splitvariables largest tree
#If tol = null, only splitvariables taken into account
M1 <- function (observeddata, treedata, Y, X, trees, tol){

  #check whether there are any categorical predictors
  #if so, this measure with splitvariables can not be used.
  Xclass<- sapply(1:length(X), function(i) class(observeddata[,X[i]] ) == "integer"|class(observeddata[,X[i]] ) == "numeric")

  if(!is.null(tol)){
    if("FALSE" %in% Xclass){
      cat("This measure only works on numerical splitvariables (class integer or numeric)")
      return(NULL)
    }
  }

  pamtrees <- lapply(1:length(trees), function (i) pamtree(observeddata,treedata[[i]], Y[i],trees[[i]]))

  simmatrix <- matrix(c(0), length(trees), length(trees))
  splits <- lapply(1:length(trees), function(i) splitv(pamtrees[[i]], tol))
  s <- sapply(1:length(trees), function (k) sapply(k:length(trees), function (l) sim(splits1=splits[[k]], splits2=splits[[l]], X=X, tol=tol)))

  #replace naNs -- trees without splits should have similarity of 1 with each other
  for (i in 1:length(trees)){
    si<- s[[i]]
    si[is.nan(si)] <- 1
    simmatrix[i, c(i:length(trees))] <- si
  }

  ind <- lower.tri(simmatrix)   #make matrix symmetric
  simmatrix[ind] <- t(simmatrix)[ind]
  return(simmatrix)
}

### Measure 2: Paths
M2<- function(observeddata, treedata, Y, trees, weight){
  if (is.null(weight)){weight=0}

  pamtrees <- lapply(1:length(trees), function (i) pamtree(observeddata,treedata[[i]], Y[i],trees[[i]]))

  n <- length(pamtrees)
  simmatrix <- matrix(c(0), n, n)
  subs <- sapply (1:n, function (k) returnsubpaths(pamtrees[[k]])) #split up each path into all subpaths

  dis <- matrix(c(0), length(pamtrees), length(pamtrees))
  d <- sapply(1:n, function (s) sapply(s:n, function (j) dissim(subs[[s]], subs[[j]], weight) ))

  for (i in 1:n){
    dis[i, c(i:n)] <- d[[i]]
  }
  ind <- lower.tri(dis)   #make matrix symmetric
  dis[ind] <- t(dis)[ind]
  sim <- 1 - round(dis,4)
  return(sim)
}


####### Measure 3: classification labels agreement
M3<- function (observeddata, treedata, Y, trees, sameobs){
  pamtrees<- lapply(1:length(trees), function (i) pamtree(observeddata,treedata[[i]], Y[i],trees[[i]]))
  n <- length(pamtrees)
  sim <- matrix(0, length(pamtrees), length(pamtrees))

  # if treedata contains same observations as fulldata, then use the training data to evaluate agreement,
  #otherwise use fulldata
  if(sameobs==TRUE){
    s <- sapply(1:n, function (s) sapply(s:n, function (j) mean(pamtrees[[s]]$predresptrain==pamtrees[[j]]$predresptrain)))
  }else{
    s <- sapply(1:n, function (s) sapply(s:n, function (j) mean(pamtrees[[s]]$predresp==pamtrees[[j]]$predresp)))
  }

   for (i in 1:n){
    sim[i,c(i:n)] <- s[[i]]
  }

  ind <- lower.tri(sim)   #make matrix symmetric
  sim[ind] <- t(sim)[ind]
  return(sim)
}

###### Measure 4: Partition metric ################
M4<- function (observeddata, treedata, Y, trees, sameobs){
  pamtrees <- lapply(1:length(trees), function (i) pamtree(observeddata,treedata[[i]], Y[i],trees[[i]]))

  n <- length(trees)
  if(sameobs==TRUE){
    par<- lapply(1:n, function (s) part(treedata[[s]], pamtrees[[s]]$prednodetrain))
  }else{
    par<- lapply(1:n, function (s) part(treedata[[s]], pamtrees[[s]]$prednode))
  }

  par<- lapply(1:n, function (s) part(treedata[[s]], pamtrees[[s]]$prednode))
  no_cores <- detectCores() - 1
  cl <- makeCluster(no_cores)
  clusterExport(cl, c("metr", "par", "treedata", "n"), envir=environment())
  si <- parSapply(cl, 1:n, function (s) sapply (s:n, function (j) metr(par[[s]], par[[j]], treedata[[s]])))
  stopCluster(cl)

  sim <- matrix(0, length(trees), length(trees))
  for (i in 1:n){
    sim[i, c(i:n)] <- si[[i]]
  }
  ind <- lower.tri(sim)   #make matrix symmetric
  sim[ind] <- t(sim)[ind]
  return(sim)
}



### set of splitvariables and splitpoints and the prediction of a leaf
M6 <- function (observeddata, treedata, Y, X, trees, tol){

  pamtrees <- lapply(1:length(trees), function (i) pamtree(observeddata,treedata[[i]], Y[i],trees[[i]]))

  n <- length(pamtrees)
  simmatrix <- matrix(c(0), n, n)

  s <- lapply(1:n, function (k) splitvsets(pamtrees[[k]]$path1))
  s0<- lapply(1:n, function (k) splitvsets(pamtrees[[k]]$path0))

  for (k in 1:n){
    if(class(treedata[[1]][,Y[1]]) != "factor" ) {
      treedata[[k]][,Y[k]] <- as.factor(treedata[[k]][,Y[k]])
    }else{
      treedata[[k]] = treedata[[k]]
    }#check whether why is a factor, if not-- make it a factor
  }

  best<- lapply(1:n, function(k) (sum(treedata[[k]][,Y[k]] == levels(treedata[[k]][,Y[k]])[2])/nrow(treedata[[k]] ))>0.5)

  sim <- sapply(1:n, function (k) sapply(k:n, function (l) simsets(paths1=s[[k]], paths2=s[[l]], paths01=s0[[k]], paths02=s0[[l]], X=X, tol=tol, best1=best[[k]],best2=best[[l]])))

  for (i in 1:n){
    si<- sim[[i]]
    si[is.nan(si)] <- 1
    simmatrix[i, c(i:n)] <- si
  }

  ind <- lower.tri(simmatrix)   #make matrix symmetric
  simmatrix[ind] <- t(simmatrix)[ind]
  return(simmatrix)
}


M5 <- function(observeddata, treedata, Y, X, trees, sameobs){
  pamtrees <- lapply(1:length(trees), function (i) pamtree(observeddata,treedata[[i]], Y[i],trees[[i]]))
  n <-length(pamtrees)
  s <- matrix(0, length(pamtrees), length(pamtrees))
  di <- lapply(1:length(pamtrees), function (i) disjnorm(pamtrees[[i]], trees[[i]],observeddata, treedata[[i]] ,X, Y[i], sameobs))

  no_cores <- detectCores() - 1
  cl <- makeCluster(no_cores)
  clusterExport(cl, c("dis", "di", "n", "pamtrees" ), envir=environment())

  if(sameobs==TRUE){
    si <- parSapply(cl, 1:n, function (i) sapply (i:n, function (j) dis(di[[i]], di[[j]], pamtrees[[i]]$predresptrain, pamtrees[[j]]$predresptrain)))
  }
  else{
    si <- parSapply(cl, 1:n, function (i) sapply (i:n, function (j) dis(di[[i]], di[[j]], pamtrees[[i]]$predresp, pamtrees[[j]]$predresp)))
  }

  si <- parSapply(cl, 1:n, function (i) sapply (i:n, function (j) dis(di[[i]], di[[j]], pamtrees[[i]]$predresp, pamtrees[[j]]$predresp)))
  stopCluster(cl)

  for (i in 1:n){
    s[i, c(i:n)] <- si[[i]]
  }

  ind <- lower.tri(s)   #make matrix symmetric
  s[ind] <- t(s)[ind]
  s<- round(s, digits=3)
  return(s)
}






## Function to turn each tree into a pam tree, containing the set of rules, the predictions on the full dataset and the nodes
## observeddata is the full dataset, treedata is the dataset for the current tree, Y is a vector with the outcome for each tree and tree are the partytrees

pamtree<- function(observeddata,treedata,Y,tree){
  if(length(tree) > 2){   ## Check whether there was a split
    paths <- listrulesparty(x=tree)   # lists all the paths from root to leave
    prednode <- predict(tree, newdata = observeddata, type = "node")  #predicts node for every row of full data
    if(class(treedata[,Y]) != "factor" ) {treedata[,Y] <- as.factor(treedata[,Y])}  #check whether y is a factor, if not-- make it a factor
    if(class(observeddata[,Y]) != "factor" ) {observeddata[,Y] <- as.factor(observeddata[,Y])}  #check whether y is a factor, if not-- make it a factor

    #if(class(tree)[1]== "glmtree"){
      predresp <- predict(tree, newdata= observeddata, type="response")  #predicts response for every row of full data
     # predresp[predresp<0.5] <- levels(observeddata[,Y])[1]
    #  predresp[predresp!=levels(observeddata[,Y])[1]] <- levels(observeddata[,Y])[2]
      predresp <- factor(predresp, levels=c(levels(observeddata[,Y])[1], levels(observeddata[,Y])[2]))
    #}else{
     # predresp <- predict(tree, newdata = observeddata, type="prob")  #predicts response for every row of full data
      #predresp <- predresp[,1]
      #predresp[predresp<0.5]<- levels(observeddata[,Y])[1]
      #predresp[predresp!= levels(observeddata[,Y])[1]]<- levels(observeddata[,Y])[2]
      #predresp <- factor(predresp, levels=c(levels(observeddata[,Y])[1], levels(observeddata[,Y])[2]))
    #}


    #if(class(tree)[1]== "glmtree"){
      predresptrain <- predict(tree, newdata = treedata, type="response")  #predicts response for every row of full data
     # predresptrain[predresptrain<0.5] <- levels(observeddata[,Y])[1]
    #  predresptrain[predresptrain!=levels(observeddata[,Y])[1]] <- levels(observeddata[,Y])[2]
      predresptrain <- factor(predresptrain, levels=c(levels(observeddata[,Y])[1], levels(observeddata[,Y])[2]))
    #}else{
    #  predresptrain <- predict(tree, newdata = treedata, type="prob")  #predicts response for every row of full data
    #  predresptrain <- predresptrain[,1]
    #  predresptrain[predresptrain<0.5]<- levels(observeddata[,Y])[1]
    #  predresptrain[predresptrain!= levels(observeddata[,Y])[1]]<- levels(observeddata[,Y])[2]
    #  predresptrain <- factor(predresptrain, levels=c(levels(observeddata[,Y])[1], levels(observeddata[,Y])[2]))
    #}

    prednodetrain <- predict(tree, newdata = treedata, type = "node")  #predicts node for every row of tree data

    frame <- matrix(c(0), length(unique(prednodetrain)), 2) #create matrix with one row for each node value
    frame[, 1] <- sort(unique(prednodetrain))
    frame[, 2] <- sapply(sort(unique(prednodetrain)), function(k) levels(predresptrain[prednodetrain == k][1])[predresptrain[prednodetrain == k][1]])  # check the predicted response for every node


    #the paths that lead to a response of the second level of y
    path1 <- paths[frame[, 2] == levels(observeddata[,Y])[2]]
    path0 <- paths[frame[, 2] == levels(observeddata[,Y])[1]]

    paths <- sapply(1:length(paths), function (k) strsplit(paths[k], " & "))  #split rules with multiple conditions in substrings
    path1 <- sapply(1:length(path1), function (k) strsplit(path1[k], " & "))
    path0 <- sapply(1:length(path0), function (k) strsplit(path0[k], " & "))

    #if there was no split
  }else{
    paths<- NULL
    path1<-NULL
    path0<- NULL

    tree <- list(numeric(0), numeric(0))
    if(class(treedata[,Y]) != "factor" ) {treedata[,Y] <- as.factor(treedata[,Y])}
    y <- treedata[, Y]

    #check what class is most prevalent
    if(sum(y == levels(y)[1], na.rm=T) > sum(y == levels(y)[2], na.rm=T)){
      g1 <- levels(y)[1]
    } else{
      g1<- levels(y)[2]
    }
    predresp <- rep(g1, length(y))
    predresp<- factor(predresp, levels=levels(treedata[,Y]))
    prednode <- rep(1, length(y))
  }


  value <- list(pamtree = paths, path0=path0, path1= path1, prednode = prednode, predresp=predresp,prednodetrain=prednodetrain, predresptrain=predresptrain)
  attr(value, "class") <- "pamtree"
  return(value)
}

#### grow party tree (for turning ranger/randomforest tree into partytree)
grow.party.tree <- function(party.tree, ranger.tree, data, factor.terms.index, currentNodeNumber) {
  node <- ranger.tree[ranger.tree$nodeID == currentNodeNumber, ]
  
  factor.terms.index.left <- factor.terms.index
  factor.terms.index.right <- factor.terms.index
  
  # Create individual node
  if (node$terminal == TRUE) {
    newNode <- list(id = node$nodeID)
  } else {
    dataclass <- class(data[[node$splitvarName]])
    if ("numeric" %in% dataclass || "ordered" %in% dataclass) {
      newNode <- list(id = node$nodeID, split = partysplit(varid = as.integer(node$splitvarID + 1), breaks = as.numeric(node$splitval)), kids = c(as.integer(node$leftChild), as.integer(node$rightChild)))
    } else {
      index <- factor.terms.index[[node$splitvarName]]
      index[as.integer(unlist(strsplit(node$splitval, ',')))] = 2L
      newNode <- list(id = node$nodeID,
                      split = partysplit(
                        varid = as.integer(node$splitvarID + 1),
                        index = index
                      ), 
                      kids = c(as.integer(node$leftChild), as.integer(node$rightChild)))
      factor.terms.index.left[[node$splitvarName]] <- replace(index, index==2L, NA)
      factor.terms.index.right[[node$splitvarName]] <- replace(replace(index, index==1L, NA), index==2L, 1L)
    }
  }
  
  # Traverse tree recursively
  if (node$terminal == FALSE) {
    leftChildren <- grow.party.tree(party.tree, ranger.tree, data, factor.terms.index.left, node$leftChild)
    rightChildren <- grow.party.tree(party.tree, ranger.tree, data, factor.terms.index.right, node$rightChild)
    
    party.tree <- c(party.tree, leftChildren, rightChildren)
  }
  
  # Add newly created node to list of nodes
  party.tree <- c(party.tree, list(newNode))
  
  party.tree
}


generic2party <- function(data, generic.tree, inbag, formula, weights) {
  response <- all.vars(formula)[1]
  terms <- terms(formula, data = data)
  factor.terms <- all.vars(terms)[-1]
  
  factor.terms.index <- list()
  for(factor.term in factor.terms) {
    factor.terms.index[[factor.term]] <- rep(1L, length(levels(data[[factor.term]])))
  }
  
  data <- data[complete.cases(data), c(all.vars(terms)[-1], response)]
  
  data <- as.data.frame(lapply(data, rep, inbag))
  
  if (is.null(weights)) {
    weights <- rep(1L, nrow(data)) 
  }
  
  nodelist = list()
  nodelist <- grow.party.tree(nodelist, generic.tree, data, factor.terms.index, 0)
  
  nodes <- as.partynode(nodelist)
  fitted <- fitted_node(nodes, data = data)
  
  tree <- party(nodes,
                data = data, 
                fitted = data.frame("(fitted)" = fitted,
                                    "(response)" = data[[response]],
                                    "(weights)" = weights,
                                    check.names = FALSE),
                terms = terms(formula, data = data)
  )
  as.constparty(tree)
}

ranger2party <- function(data, ranger.forest, treeNumber, weights = NULL) {
  if (!exists("inbag.counts", where=ranger.forest)) {
    stop("Run ranger with the keep.inbag=T parameter")
  }
  
  ranger.tree <- treeInfo(ranger.forest, tree = treeNumber)
  formula <- formula(ranger.forest$call[[2]])
  inbag <- ranger.forest$inbag.counts[[treeNumber]]
  
  generic2party(data, ranger.tree, inbag, formula, weights)
}

randomForest2party <- function(data, randomForest.forest, treeNumber, weights = NULL) {
  if (!exists("inbag", where=randomForest.forest)) {
    stop("Run randomForest with the keep.inbag=T parameter")
  }
  
  randomForest.tree.without.labels <- data.frame(getTree(randomForest.forest, k = treeNumber, labelVar = F))
  randomForest.tree.with.labels <- data.frame(getTree(randomForest.forest, k = treeNumber, labelVar = T))
  
  # Convert randomForest format to Ranger format  
  colnames(randomForest.tree.with.labels) <- c("leftChild", "rightChild", "splitvarName", "splitval", "terminal", "prediction")
  colnames(randomForest.tree.without.labels) <- c("leftChild", "rightChild", "splitvarID", "splitval", "terminal", "prediction")
  nodeID <- 0:(nrow(randomForest.tree.with.labels) -1)
  leftChild <- randomForest.tree.with.labels$leftChild - 1L
  rightChild <- randomForest.tree.with.labels$rightChild - 1L
  splitvarID <- randomForest.tree.without.labels$splitvarID - 1L
  splitvarName <- as.character(randomForest.tree.with.labels$splitvarName)
  splitval <- randomForest.tree.with.labels$splitval
  terminal <- ifelse(randomForest.tree.without.labels$terminal == -1, TRUE, FALSE)
  is.na(rightChild) <- is.na(splitvarID) <- is.na(leftChild) <- is.na(splitval) <- terminal
  prediction <- randomForest.tree.with.labels$prediction
  generic.tree <- data.frame(nodeID, leftChild, rightChild, splitvarID, splitvarName, splitval, terminal, prediction)
  
  idx.unordered <- apply(array(splitvarName), 1, function(x) { !("ordered" %in% class(data[[x]]) || "numeric" %in% class(data[[x]]))})
  idx.unordered[terminal] <- FALSE
  
  if (any(idx.unordered)) {
    if (any(generic.tree$splitval[idx.unordered] > (2^31 - 1))) {
      warning("Unordered splitting levels can only be shown for up to 31 levels.")
      generic.tree$splitval[idx.unordered] <- NA
    } else {
      generic.tree$splitval[idx.unordered] <- sapply(generic.tree$splitval[idx.unordered], function(x) {
        paste(which(as.logical(intToBits(2^31-1-x))), collapse = ",")
      })
    }
  }  
  
  formula <- formula(randomForest.forest$call[[2]])
  
  inbag <- randomForest.forest$inbag[, treeNumber]
  
  generic2party(data, generic.tree, inbag, formula, weights)
}





### partykit:::.list.rules.party
listrulesparty <- function (x, i = NULL, ...)
{
  if (is.null(i))
    i <- nodeids(x, terminal = TRUE)
  if (length(i) > 1) {
    ret <- sapply(i, listrulesparty, x = x)
    names(ret) <- if (is.character(i))
      i
    else names(x)[i]
    return(ret)
  }
  if (is.character(i) && !is.null(names(x)))
    i <- which(names(x) %in% i)
  stopifnot(length(i) == 1 & is.numeric(i))
  stopifnot(i <= length(x) & i >= 1)
  i <- as.integer(i)
  dat <- data_party(x, i)
  if (!is.null(x$fitted)) {
    findx <- which("(fitted)" == names(dat))[1]
    fit <- dat[, findx:ncol(dat), drop = FALSE]
    dat <- dat[, -(findx:ncol(dat)), drop = FALSE]
    if (ncol(dat) == 0)
      dat <- x$data
  }
  else {
    fit <- NULL
    dat <- x$data
  }
  rule <- c()
  recFun <- function(node) {
    if (id_node(node) == i)
      return(NULL)
    kid <- sapply(kids_node(node), id_node)
    whichkid <- max(which(kid <= i))
    split <- split_node(node)
    ivar <- varid_split(split)
    svar <- names(dat)[ivar]
    index <- index_split(split)

    if (is.factor(dat[, svar])) {
      if (is.null(index))
        index <- ((1:nlevels(dat[, svar])) > breaks_split(split)) +
          1
      slevels <- levels(dat[, svar])[index == whichkid]
      srule <- paste(svar, " %in% c(\"", paste(slevels,
                                               collapse = "\", \"", sep = ""), "\")", sep = "")
    }
    else {
      if (is.null(index))
        index <- 1:length(kid)
      breaks <- cbind(c(-Inf, breaks_split(split)), c(breaks_split(split),
                                                      Inf))
      sbreak <- breaks[index == whichkid, ]
      right <- right_split(split)
      srule <- c()
      if (is.finite(sbreak[1]))
        srule <- c(srule, paste(svar, ifelse(right, ">",
                                             ">="), sbreak[1]))
      if (is.finite(sbreak[2]))
        srule <- c(srule, paste(svar, ifelse(right, "<=",
                                             "<"), sbreak[2]))
      srule <- paste(srule, collapse = " & ")
    }
    rule <<- c(rule, srule)
    return(recFun(node[[whichkid]]))
  }
  node <- recFun(node_party(x))
  paste(rule, collapse = " & ")
}


#### Subfunctions M1 ######
#get splitvariables and splitpoints
splitv <- function (tree, tol){
  paths <- tree$pamtree
  # check whether there was a split
  if(length(paths) > 0){
    pathsw <- paths
    leaves <- length(paths)
    l <- sapply(1:leaves, function (k) length(paths[[k]]))
    spaths <- lapply(1:leaves, function(k) sub(" <=.", ':', paths[[k]]))
    spaths <- lapply(1:leaves, function(k) sub(" <.", ':', spaths[[k]]))
    spaths <- lapply(1:leaves, function(k) sub(" >=.", ':', spaths[[k]]))
    spaths <- lapply(1:leaves, function(k) sub(" >.", ':', spaths[[k]]))
    spaths <- lapply(1:leaves, function(k) sub(" %in%.*", '', spaths[[k]]))

    splitvarsa <- unique(unlist(spaths))
    splitvars <- sub(":.*", '', splitvarsa)
    if(!is.null(tol)){
      splitvarsa <- sub(".*:", '', splitvarsa)
      splitvarsa <- as.numeric(splitvarsa)
    } else{
      splitvarsa<- NULL
    }
    nsplitvar1 <- length(splitvars)       # number of splits in tree i
  }else{
    splitvars <- NULL
    splitvarsa <- NULL
    nsplitvar1 <- 0
  }
  return(list(splitvars = splitvars, splitpoints = splitvarsa, nsplits = nsplitvar1))
}


#calculate jaccard index
sim<- function (splits1, splits2, X, tol){
  ### Only predictors
  if(is.null(tol)){
    common1 <- length(splits1$splitvars[pmatch(splits1$splitvars, splits2$splitvars, nomatch = 0)]) #pmatch: no doubles
    total1 <- splits1$nsplits + splits2$nsplits - common1
    Jaccard <- common1 / total1
  }
  ### Also splitpoints
  if(!is.null(tol)){
    # Create matrix That shows for each variable of the splits1 whether it is equal to each variable in splits2
    same <- matrix(c(0), length(splits1[[1]]), length(splits2[[1]]))

    if(length(same > 0)){
      for (i in 1:length(splits1[[1]])){
        for(j in 1:length(splits2[[1]])){
          same[i, j] <- splits1[[1]][[i]] == splits2[[1]][[j]]
        }
      }

      # For those variables that are the same in both trees, put splitpoints of tree 2 in the matrix
      splitpoints <- matrix(c(rep(splits2$splitpoints, nrow(same))), nrow(same), ncol(same), byrow=T)
      s <- c(splitpoints)
      sm <- c(same)
      s[sm == 0] <- NA
      splitpoints <- matrix(s, nrow(same), ncol(same), byrow=F)

      # Get the right tolerance for each variables in splits1
      tsa <- splits1$splitvars
      t <- match(tsa,X)
      correct<- matrix(c(0), nrow(same), ncol(same))
      # Look whether splitpoint of splits2 is within tolerance zone of splits1
      for (i in 1:length(splits1$splitvars)){
        d <- as.numeric(tol[t[i]])
        correct[i, ] <- findInterval(splitpoints[i, ], c(as.numeric(splits1$splitpoints[i]) - d, as.numeric(splits1[[2]][i]) + d) ) == 1
      }

      correct[is.na(correct)] <- 0
      g <- graph_from_incidence_matrix(correct)
      common<- max_bipartite_match(g)$matching_size

      total<- splits1[[3]] + splits2[[3]] - common
      Jaccard<- common / total
    } else{
      if(length(splits1[[1]]) == 0 & length(splits2[[1]]) == 0){
        Jaccard <- 1
      } else{Jaccard <- 0}
    }
  }
  return(Jaccard)

}

# function returns all the subpaths, takes away splitpoints and removes direction of last split of each path.
# then returns only unique subpaths
## subfunction M2
returnsubpaths <- function(tree){
  paths <- tree$pamtree

  if(length(paths) > 0){
    leaves<- length(paths)
    l<- sapply(1:leaves, function (k) length(paths[[k]]))

    lastpaths <- lapply(1:leaves, function(k) sub("<.*", '', paths[[k]][l[k]])) # remove direction and splitpoint last attribut of path
    lastpaths <- lapply(1:leaves, function(k) sub(">.*", '', lastpaths[[k]]))

    # place it back in paths
    for(j in 1:leaves){
      paths[[j]][l[j]]<- lastpaths[j]
    }

    paths <- lapply(1:leaves, function(k) sub("<.*", '-', paths[[k]]))   #replace splitpoints with - or +
    paths <- lapply(1:leaves, function(k) sub(">.*", '+', paths[[k]]))
    upaths <- unique(paths)

    subpaths <- list(0)
    for(j in 1:length(upaths)){
      d<- list(0)
      d[[1]] <- upaths[[j]]
      if(length(upaths[[j]]) > 1){    #check whether more than one split (otherwise no subpaths and d has just one element)
        for (i in 2:length(upaths[[j]])){
          d[[i]] <- d[[i - 1]][- length(d[[i - 1]])]   #Split each path up into all subpaths
        }
      }
      subpaths[[j]] <- d
    }
    subpaths <- unlist(subpaths, recursive=FALSE)
    lastsubpaths <- lapply(1:length(subpaths), function(k) gsub("[[:punct:]]", '', subpaths[[k]][length(subpaths[[k]])]))  # remove punctuation from last attribute of each subpath
    for(j in 1:length(subpaths)){
      subpaths[[j]][length(subpaths[[j]])] <- lastsubpaths[j]
    }

    subpaths<- unique(subpaths)
  }else{subpaths<- NULL}
  return(subpaths)
}


#calculates number of distinct subpaths in two sets of subpaths
dissim<- function (subs1,subs2,weights){
  l <- sapply(1:length(subs1), function (k) length(subs1[[k]]))  # length of each subpath of each path
  d <- c(0)
  l2 <- sapply(1:length(subs2), function (k) length(subs2[[k]])) #length of each subpath of each path

  for (k in 1:(max(max(l), max(l2)))){  #iterate until longest subpath of the two trees
    a <- as.numeric(subs1[l == k] %in% subs2[l2 == k])  # number of subpaths of tree i in j
    b <- as.numeric(subs2[l2 == k] %in% subs1[l == k])  # number of subpaths of tree j in i

    if(weights==0){d[k] <- (length(a) + length(b)) - (sum(a) + sum(b))}   # if weighs 0 every dissimilar subpath weighted equally
    if(weights==1){d[k] <- 1 / k * ((length(a) + length(b)) - (sum(a) + sum(b)))} # if weights 1 every dissimilar subpath weighted by 1/length subpath
  }

  if(weights == 0){dis <- sum(d) / (length(l[l > 0]) + length(l2[l2 > 0]))}  # divide #dissimilar subpaths by maximum dissimilar subpaths
  wl <- sum(1 / l)
  wl2 <- sum(1 / l2)
  wl[is.infinite(wl)] <- 0
  wl2[is.infinite(wl2)] <- 0
  if(weights == 1){dis <- sum(d) / (wl+wl2)}  #divide weighted #dissimilar subpaths by maximum weighted # dissimilar subpaths

  dis[is.na(dis)] <- 0
  return(dis)
}



### Subfunctions M4
part <- function (data, tree1){
  t1 <- matrix(c(0), nrow(data), nrow(data))
  for(i in 1:nrow(data) - 1){
    t1[i,] <- tree1[i] == tree1  # for each observation with each other observation: Same leaf?
  }
  return(t1)
}

metr<- function (t1, t2, data){
  ind <- upper.tri(t1)
  part<- sum(t1[ind] == t2[ind]) / choose(nrow(data), 2)
  return(part)
}



#### Subfunctions M6
# function to get back the splitvariables and slit points for each path
splitvsets <- function (paths){
  splitvars<- list(length(paths))
  splitvarsa<- list(length(paths))
  nsplitvar1<- list(length(paths))

  #paths <- trees[[1]]
  if(length(paths) > 0){
    for(i in 1:length(paths)){
      pathsw <- paths[[i]]

      # splitvarsa1 <- unique(unlist(spaths))
      splitvarsa1<- unique(unlist(pathsw))
      splitvars1<- sub("\\-.*$", '', splitvarsa1)
      splitvars[[i]]<- sub("\\d.*$", '', splitvars1)

      splitvarsa1 <- sub(".*=", '', pathsw)
      splitvarsa1 <- sub(".*>", '', splitvarsa1)
      splitvarsa1 <- sub(".*<", '', splitvarsa1)
      splitvarsa[[i]] <- c(as.numeric(splitvarsa1))

    }
  }else{
    splitvars <- NULL
    splitvarsa <- NULL
    #nsplitvar1 <- 0
  }
  return(list(splitvars = splitvars, splitvarsa = splitvarsa))
}


simsets<- function (paths1, paths2, paths01, paths02, X, tol,best1,best2){


  if(length(paths1[[1]])&length(paths2[[1]])>0) {
    J1 <- JaccardPaths(paths1, paths2, tol,X)
    J0 <- JaccardPaths(paths01, paths02,tol,X)
    sim<- (J1+J0)/2

  } else{
    if(length(paths1[[1]]) == 0 & length(paths2[[1]]) == 0){
      if(best1==best2){
        sim <- 1
      } else{sim<-0}

    } else{sim <- 0}
  }
  return(sim)
}

JaccardPaths <- function(paths1, paths2, tol,X){
  Jaccard<- matrix(c(0),length(paths1[[1]]),length(paths2[[1]]))

  ## for each path in tree 1 and for each path in tree 2
  for (i in 1:length(paths1[[1]])){
    for (j in 1:length(paths2[[1]])){
      same <- matrix(c(0), length(paths1[[1]][[i]]), length(paths2[[1]][[j]]))

      # for the path check whether each variable has a match in the other path
      if(length(same > 0)){
        for (k in 1:length(paths1[[1]][[i]])){
          for(m in 1:length(paths2[[1]][[j]])){
            same[k, m] <- paths1[[1]][[i]][[k]] == paths2[[1]][[j]][[m]]
          }
        }


        ### In case splitpoints are taken into account
        if (!is.null(tol)){
          # For those variables that are the same in both trees, put splitpoints in the matrix
          splitpoints <- matrix(c(rep(paths2[[2]][[j]], nrow(same))), nrow(same), ncol(same), byrow=T)
          s <- c(splitpoints)
          sm <- c(same)
          s[sm == 0] <- NA
          splitpoints <- matrix(s, nrow(same), ncol(same), byrow=F)

          tsa<- paths1[[1]][[i]]
          tsa<- sub(" <=.", '', tsa)
          tsa<- sub(" <.", '', tsa)
          tsa<- sub(" >=.", '', tsa)
          tsa<- sub(" >.", '', tsa)
          t <- match(tsa,X)
          correct<- matrix(c(0), nrow(same), ncol(same))
          # Look whether splitpoint of splits2 is within tolerance zone of splits1
          for (h in 1:length(paths1[[1]][[i]])){
            d <- as.numeric(tol[t[h]])
            correct[h, ] <- findInterval(splitpoints[h, ], c(as.numeric(paths1[[2]][[i]][h]) - d, as.numeric(paths1[[2]][[i]][h]) + d) ) == 1
          }
        }else{
          correct<- same
        }

        correct[is.na(correct)] <- 0
        g <- graph_from_incidence_matrix(correct)
        common<- max_bipartite_match(g)$matching_size

        total<- length(paths1[[1]][[i]]) + length(paths2[[2]][[j]]) - common
        Jaccard[i,j]<- common / total
      }else{
        if(length(paths1[[1]][[i]]) == 0 & length(paths2[[1]][[j]]) == 0){
          Jaccard[i,j] <- 1
        } else{Jaccard[i,j] <- 0}
      }
    }
  }

  g <- graph_from_incidence_matrix(Jaccard, weighted=TRUE)
  common<- max_bipartite_match(g)$matching_weight
  sim<- common/min(length(paths1[[1]]), length(paths2[[2]]))
  return(sim)
}


##### sUBFUNCTIONS M5
##############################################################
dis <- function (tree1,tree2, predresp1, predresp2){

  if( ! is.null(tree1) & ! is.null(tree2)){
    common <- matrix(c(0), length(tree1), length(tree2))

    # how many matches in each rule
    for(i in 1:length(tree1)){
      for(j in 1: length(tree2)){
        common[i, j] <- length(unlist(tree1[i])[pmatch(unlist(tree1[i]), unlist(tree2[j]), nomatch = 0)])
      }
    }
    rows <- apply(common, 2, max)
    cols <- apply(common, 1, max)


    #common rules are those that have all their elementary conjuncts in common
    #calculate jaccard
    common <- min(sum(rows == length(tree1[[1]])), sum(cols == length(tree1[[1]])))
    total <- length(tree1) + length(tree2) - common
    s <- common / total

  }else if(is.null(tree1) & ! is.null(tree2) | ! is.null(tree1) & is.null(tree2)){
    s <- 0
  }else{
    #check what prediction was made by the trivial rule
    if(predresp1[1] == predresp2[1]){
      s <- 1
    } else{s<-0}

  }

  return(s)
}



disjnorm <- function (pamtree, tree,observeddata, treedata,X, Y, sameobs){
  path <- pamtree$path1    #Paths that lead to leaf with predicted class 1

  if(sameobs==TRUE){
    predresp<- pamtree$predresptrain
  }else{
    predresp<- pamtree$predresp
  }


  if(length(path) > 0){
    leaves <- length(path)  # number of leaves
    l <- sapply(1:leaves, function (k) length(path[[k]]))     # for each path number of elements
    levels <- matrix(c(0), length(X), 2)

    for (i in 1:length(X)){
      levels[i, ] <- levels(observeddata[, X[i]])
    }

    levels <- cbind(X, levels)
    npaths <-path
    spaths <- lapply(1:leaves, function(k) sapply(1:l[k], function (j) sub(" %in%.*", '', path[[k]][j])))

    ##
    for(i in 1:length(X)){
      npaths <- lapply(1:leaves, function(k) sapply(1:l[k], function (j) sub(paste(levels[i,2]), '-', npaths[[k]][j])))
    }

    for(i in 1:length(X)){
      npaths <- lapply(1:leaves, function(k) sapply(1:l[k], function (j) sub(paste(levels[i,3]), '+', npaths[[k]][j])))
    }

    for(i in 1:length(X)){
      signs <- lapply(1:leaves, function (i) gsub("[^-+]+", "", npaths[[i]]))
    }

    paths <- lapply(1:leaves, function (i) paste(spaths[[i]], signs[[i]]))
    b <- sapply (1:length(spaths), function (s) X[pmatch(X, spaths[[s]], nomatch = 0) == 0])

    if(length(b) > 0){
      if(is.list(b)){
        for(i in 1:length(b)){
          now <- b[[i]]
          nowe <- now[now != ""]
          nowe <- sort(nowe)
          # Add those variables in plus and minus form to those paths
          if(length(nowe > 0)){
            a <- matrix(c(0), 2, length(nowe))
            a[1, ] <- c(paste(nowe, '+', sep=' '))
            a[2, ] <- c(paste(nowe, '-', sep=' '))
            grid <- do.call(expand.grid, split(a, col(a)))

            for (k in 1:nrow(grid)){
              paths <- c(paths, list(unlist(c(paths[i], paste(unlist(grid[k,c(1:ncol(grid))]))))))
            }
          } else {paths <- paths}
        }
      }else{
        now <- b
        nowe <- now[now != ""]
        nowe <- sort(nowe)
        # Add those variables in plus and minus form to those paths
        if(length(nowe > 0)){
          a <- matrix(c(0), 2, length(nowe))
          a[1, ] <- c(paste(nowe, '+', sep=' '))
          a[2, ] <- c(paste(nowe, '-', sep=' '))
          grid <- do.call(expand.grid, split(a, col(a)))
          paths <- lapply (1:nrow(grid), function (k) c(paths[[1]], paste(unlist(grid[k,c(1:ncol(grid))]))))
        } else {
          paths <- paths
        }
      }

    } else{paths <- paths}


    l <- sapply(1:length(paths), function (k) length(paths[[k]]))
    paths <- unique(paths)
    paths <- paths[lengths(paths) == max(l)]

  }else{   ### if tree only root
    if(predresp[1] == levels(observeddata[,Y])[1]){
      paths <- NULL
    }else{
      a <- matrix(c(0), 2, length(X))
      a[1, ] <- c(paste(X, '+', sep=' '))
      a[2, ] <- c(paste(X, '-', sep=' '))
      grid <- do.call(expand.grid, split(a, col(a)))
      paths <- lapply (1:nrow(grid), function (k) paste(unlist(grid[k, c(1:ncol(grid))])))
    }
  }
  return(paths)
}




