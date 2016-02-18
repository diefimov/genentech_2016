library(data.table)
library(pROC)
library(bit64)
#library(xgboost)
options(stringsAsFactors = FALSE)
path.wd <- getwd()
all.noexport <- character(0)


sna <- function(x){
  r<-sum(is.na(x))
  r
}
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

colRank <- function(X) apply(X, 2, rank)
colMedian <- function(X) apply(X, 2, median)
colMax <- function(X) apply(X, 2, max)
colMin <- function(X) apply(X, 2, min)
colSd <- function(X) apply(X, 2, sd)
mae <- function(c1,c2) {
  c1 <- as.numeric(c1)
  c2 <- as.numeric(c2)
  score <- mean( abs(c1-c2) )
  score
}
rep.row<-function(x,n){
  matrix(rep(x,each=n),nrow=n)
}
rep.col<-function(x,n){
  matrix(rep(x,each=n), ncol=n, byrow=TRUE)
}
rowMax <- function(X) apply(X, 1, max)
rowMin <- function(X) apply(X, 1, min)
rowMean <- function(X) apply(X, 1, mean,na.rm=TRUE)
rowMedian <- function(X) apply(X, 1, median,na.rm=TRUE)
rowSd <- function(X) apply(X, 1, sd)
rowMode <- function(X) apply(X, 1, Mode)
rowPaste <- function(X) apply(X, 1, paste0, collapse=" ")
rowPaste0 <- function(X) apply(X, 1, paste0, collapse="")

AUC2 <- function( true_labels, predicted  ){
  as.numeric( roc(true_labels,predicted)$auc )
}

AUC <-function (actual, predicted) {
  r <- as.numeric(rank(predicted))
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1)/2)/(n_pos *  n_neg)
  auc
}

colAUC <- function(X,tgt) apply(X, 2, AUC , tgt )

cauc <- function( pred ){
  ind <- which( bas$cv>=1 )
  pred <- pred[ind]
  pred[pred == Inf] <- NA
  pred[pred == -Inf] <- NA
  pred[is.na(pred)] <- mean(pred, na.rm=T)
  AUC( bas$is_screener[ind] ,  pred )
}
	
	
#############################################################
# tic toc
#############################################################
tic <- function(gcFirst = TRUE, type=c("elapsed", "user.self", "sys.self")) {
  type <- match.arg(type)
  assign(".type", type, envir=baseenv())
  if(gcFirst) gc(FALSE)
  tic <- proc.time()[type]         
  assign(".tic", tic, envir=baseenv())
  invisible(tic)
}

toc <- function() {
  type <- get(".type", envir=baseenv())
  toc <- proc.time()[type]
  tic <- get(".tic", envir=baseenv())
  print(toc - tic)
  invisible(toc)
}

#############################################################
# data file path
#############################################################
fn.data.file <- function(name) {
  paste(path.wd, "", name, sep="/")
}
#############################################################
# save data file
#############################################################
fn.save.data <- function(dt.name, envir = parent.frame()) {
  save(list = dt.name, 
       file = fn.data.file(paste0(dt.name, ".RData")), envir = envir)
}
#############################################################
# load saved file
#############################################################
fn.load.data <- function(dt.name, envir = parent.frame()) {
  load(fn.data.file(paste0(dt.name, ".RData")), envir = envir)
}




LIKELIHOOD <- function( DT, L , px ){
  dt <- DT[px, list( mean(tgt), length(tgt) ) , keyby="var"]
  M <- mean( DT$tgt[px], na.rm=T  )
  dt[, tgt := ( (dt$V1*dt$V2) + (L*M) ) / (L+dt$V2) ]
  tmp <- as.numeric( dt[J(DT$var)]$tgt )
  tmp
}

LIKELIHOOD_2WAY <- function( DT, L , px ){
  dt <- DT[px, list( mean(tgt), length(tgt) ) , keyby=c("var1","var2")]
  M <- mean( DT$tgt[px], na.rm=T  )
  dt[, tgt := ( (dt$V1*dt$V2) + (L*M) ) / (L+dt$V2) ]
  tmp <- dt[J(DT$var1,DT$var2)]$tgt
  tmp[ is.na(tmp)  ] <- M
  as.numeric( tmp )
}

LIKELIHOOD_3WAY <- function( DT, L , px ){
  dt <- DT[px, list( mean(tgt), length(tgt) ) , keyby=c("var1","var2","var3")]
  M <- mean( DT$tgt[px], na.rm=T  )
  dt[, tgt := ( (dt$V1*dt$V2) + (L*M) ) / (L+dt$V2) ]
  tmp <- dt[J(DT$var1,DT$var2,DT$var3)]$tgt
  tmp[ is.na(tmp)  ] <- M
  as.numeric( tmp )
}
	
LIKELIHOOD_4WAY <- function( DT, L , px ){
  dt <- DT[px, list( mean(tgt), length(tgt) ) , keyby=c("var1","var2","var3","var4")]
  M <- mean( DT$tgt[px], na.rm=T  )
  dt[, tgt := ( (dt$V1*dt$V2) + (L*M) ) / (L+dt$V2) ]
  tmp <- dt[J(DT$var1,DT$var2,DT$var3,DT$var4)]$tgt
  tmp[ is.na(tmp)  ] <- M
  as.numeric( tmp )
}

	

LIKELIHOOD_CV_MAX <- function( feat="", fname="", L=0, validate=F ){
  if( fname=="" ){
    fname <- "_"
  }
  fold1 <- rep( NA , nrow(dt1) )
  fold2 <- rep( NA , nrow(dt1) )
  fold3 <- rep( NA , nrow(dt1) )
  foldA <- rep( NA , nrow(dt1) )
  
  p1 <- which( dt1$cv==1  )
  p2 <- which( dt1$cv==2  )
  p3 <- which( dt1$cv==3  )
  pt <- which( is.na(dt1$cv)  )
  
  DT <- data.table( var=dt1[[feat]]  , tgt=dt1$tgt  )
  dt <- DT[,.N,by="var"]
  L <- median( dt$N  )
  if(L<10){
    L=10
  }
  print(paste("L",L))
  
  if( validate==F ){
    ll123 <- LIKELIHOOD( DT , L , c(p1,p2,p3)  )
    ll1   <- LIKELIHOOD( DT , L ,       p1 )
    ll2   <- LIKELIHOOD( DT , L ,       p2 )
    ll3   <- LIKELIHOOD( DT , L ,       p3 )
    ll12  <- LIKELIHOOD( DT , L , c(p1,p2) )
    ll13  <- LIKELIHOOD( DT , L , c(p1,p3) )
  }
  ll23  <- LIKELIHOOD( DT , L , c(p2,p3) )
  
  fold1[p1] <- ll23[p1]
  if( validate==F ){
    fold1[p2] <- ll3[p2]
    fold1[p3] <- ll2[p3]
    
    fold2[p2] <- ll13[p2]
    fold2[p1] <- ll3[p1]
    fold2[p3] <- ll1[p3]
    
    fold3[p3] <- ll12[p3]
    fold3[p1] <- ll2[p1]
    fold3[p2] <- ll1[p2]
    
    foldA[pt] <- ll123[pt]
    foldA[p1] <- ll23[p1]
    foldA[p2] <- ll13[p2]
    foldA[p3] <- ll12[p3]
    rm(ll1,ll2,ll3,ll12,ll13,ll123)
  }
  rm(ll23,DT,dt)
  
  if( validate==F ){
    #    dt1[, LL := NULL  ]
    dt1[, LL := fold1 ]
    dt <- dt1[, max(LL), keyby="patient_id" ]
    bas[, c(paste0("cv1_max_",fname,"_",feat)) := list(dt[J(bas$patient_id)]$V1)  ]
    
    dt1[, LL := NULL  ]
    dt1[, LL := fold2 ]
    dt <- dt1[, max(LL), keyby="patient_id" ]
    bas[, c(paste0("cv2_max_",fname,"_",feat)) := list(dt[J(bas$patient_id)]$V1)  ]
    
    dt1[, LL := NULL  ]
    dt1[, LL := fold3 ]
    dt <- dt1[, max(LL), keyby="patient_id" ]
    bas[, c(paste0("cv3_max_",fname,"_",feat)) := list(dt[J(bas$patient_id)]$V1)  ]
    
    #  dt1[, LL := NULL  ]
    dt1[, LL := foldA ]
    dt <- dt1[, max(LL), keyby="patient_id" ]
    bas[, c(paste0("cv0_max_",fname,"_",feat)) := list(dt[J(bas$patient_id)]$V1)  ]
    dt1[, LL := NULL  ];gc()
    
    ind <- which( bas$cv>=1 )
    pred <- bas[[ c(paste0("cv0_max_",fname,"_",feat)) ]][ind]
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat , AUC( bas$is_screener[ind] ,  pred ) ) )
    #  }
  }else{
    dt1[, LL := fold1 ]
    dt <- dt1[, max(LL), keyby="patient_id" ]
    dt1[, LL := NULL  ];gc()
    
    ind <- which( bas$cv==1 )
    pred <- dt[J(bas$patient_id[ind])]$V1
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat , AUC( bas$is_screener[ind] ,  pred ) ) )
  }
}

LIKELIHOOD_CV_MEAN <- function( feat="", fname="", L=0, validate=F ){
  if( fname=="" ){
    fname <- "_"
  }
  fold1 <- rep( NA , nrow(dt1) )
  fold2 <- rep( NA , nrow(dt1) )
  fold3 <- rep( NA , nrow(dt1) )
  foldA <- rep( NA , nrow(dt1) )
  
  p1 <- which( dt1$cv==1  )
  p2 <- which( dt1$cv==2  )
  p3 <- which( dt1$cv==3  )
  pt <- which( is.na(dt1$cv)  )
  
  DT <- data.table( var=dt1[[feat]]  , tgt=dt1$tgt  )
  dt <- DT[,.N,by="var"]
  L <- median( dt$N  )
  if(L<10){
    L=10
  }
  print(paste("L",L))
  
  if( validate==F ){
    ll123 <- LIKELIHOOD( DT , L , c(p1,p2,p3)  )
    ll1   <- LIKELIHOOD( DT , L ,       p1 )
    ll2   <- LIKELIHOOD( DT , L ,       p2 )
    ll3   <- LIKELIHOOD( DT , L ,       p3 )
    ll12  <- LIKELIHOOD( DT , L , c(p1,p2) )
    ll13  <- LIKELIHOOD( DT , L , c(p1,p3) )
  }
  ll23  <- LIKELIHOOD( DT , L , c(p2,p3) )

  fold1[p1] <- ll23[p1]
  if( validate==F ){
    fold1[p2] <- ll3[p2]
    fold1[p3] <- ll2[p3]
    
    fold2[p2] <- ll13[p2]
    fold2[p1] <- ll3[p1]
    fold2[p3] <- ll1[p3]
    
    fold3[p3] <- ll12[p3]
    fold3[p1] <- ll2[p1]
    fold3[p2] <- ll1[p2]
    
    foldA[pt] <- ll123[pt]
    foldA[p1] <- ll23[p1]
    foldA[p2] <- ll13[p2]
    foldA[p3] <- ll12[p3]
    rm(ll1,ll2,ll3,ll12,ll13,ll123)
  }
  rm(ll23,DT,dt)
  
  if( validate==F ){
    #    dt1[, LL := NULL  ]
    dt1[, LL := fold1 ]
    dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
    bas[, c(paste0("cv1_mean_",fname,"_",feat)) := list(dt[J(bas$patient_id)]$V1)  ]
    
    dt1[, LL := NULL  ]
    dt1[, LL := fold2 ]
    dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
    bas[, c(paste0("cv2_mean_",fname,"_",feat)) := list(dt[J(bas$patient_id)]$V1)  ]
    
    dt1[, LL := NULL  ]
    dt1[, LL := fold3 ]
    dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
    bas[, c(paste0("cv3_mean_",fname,"_",feat)) := list(dt[J(bas$patient_id)]$V1)  ]

    #  dt1[, LL := NULL  ]
    dt1[, LL := foldA ]
    dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
    bas[, c(paste0("cv0_mean_",fname,"_",feat)) := list(dt[J(bas$patient_id)]$V1)  ]
    dt1[, LL := NULL  ];gc()
    
    ind <- which( bas$cv>=1 )
    pred <- bas[[ c(paste0("cv0_mean_",fname,"_",feat)) ]][ind]
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat , AUC( bas$is_screener[ind] ,  pred ) ) )
    #  }
  }else{
    dt1[, LL := fold1 ]
    dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
    dt1[, LL := NULL  ];gc()
    
    ind <- which( bas$cv==1 )
    pred <- dt[J(bas$patient_id[ind])]$V1
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat , AUC( bas$is_screener[ind] ,  pred ) ) )
    
  }
}



LIKELIHOOD_2WAY_CV_MAX <- function( feat1="",feat2="", fname="", L=0, validate=F ){
#   feat1 = "diagnosis_code"  
#   feat2 = "primary_physician_role"  
# fname=""  
  if( fname=="" ){
    fname <- "_"
  }
  fold1 <- rep( NA , nrow(dt1) )
  fold2 <- rep( NA , nrow(dt1) )
  fold3 <- rep( NA , nrow(dt1) )
  foldA <- rep( NA , nrow(dt1) )
  
  p1 <- which( dt1$cv==1  )
  p2 <- which( dt1$cv==2  )
  p3 <- which( dt1$cv==3  )
  pt <- which( is.na(dt1$cv)  )
  
  DT <- data.table( var1=dt1[[feat1]] ,var2=dt1[[feat2]]  , tgt=dt1$tgt  )
  dt <- DT[,.N,by=c("var1","var2")]
  L <- median( dt$N  )
  rm(dt)
  if(L<4){
    L=4
  }
  print(paste("L",L));gc()
  
  if( validate==F ){
    ll123 <- LIKELIHOOD_2WAY( DT , L , c(p1,p2,p3)  )
    ll1   <- LIKELIHOOD_2WAY( DT , L ,       p1 )
    ll2   <- LIKELIHOOD_2WAY( DT , L ,       p2 )
    ll3   <- LIKELIHOOD_2WAY( DT , L ,       p3 )
    ll12  <- LIKELIHOOD_2WAY( DT , L , c(p1,p2) )
    ll13  <- LIKELIHOOD_2WAY( DT , L , c(p1,p3) )
    ll23  <- LIKELIHOOD_2WAY( DT , L , c(p2,p3) )
  }else{
    ll23  <- LIKELIHOOD_2WAY( DT , L , p2 )
  }
  rm(DT)
  
  if( validate==F ){
    fold1[p1] <- ll23[p1]
    fold1[p2] <- ll3[p2]
    fold1[p3] <- ll2[p3]
    
    fold2[p2] <- ll13[p2]
    fold2[p1] <- ll3[p1]
    fold2[p3] <- ll1[p3]
    
    fold3[p3] <- ll12[p3]
    fold3[p1] <- ll2[p1]
    fold3[p2] <- ll1[p2]
    
    foldA[pt] <- ll123[pt]
    foldA[p1] <- ll23[p1]
    foldA[p2] <- ll13[p2]
    foldA[p3] <- ll12[p3]
    rm(ll1,ll2,ll3,ll12,ll13,ll123)
  }else{
    fold1[p1] <- ll23[p1]
  }

  if( validate==F ){
    #    dt1[, LL := NULL  ]
    dt1[, LL := fold1 ]
    dt <- dt1[, max(LL), keyby="patient_id" ]
    bas[, c(paste0("cv1_max_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
    
    dt1[, LL := NULL  ]
    dt1[, LL := fold2 ]
    dt <- dt1[, max(LL), keyby="patient_id" ]
    bas[, c(paste0("cv2_max_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
    
    dt1[, LL := NULL  ]
    dt1[, LL := fold3 ]
    dt <- dt1[, max(LL), keyby="patient_id" ]
    bas[, c(paste0("cv3_max_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
    
    #  dt1[, LL := NULL  ]
    dt1[, LL := foldA ]
    dt <- dt1[, max(LL), keyby="patient_id" ]
    bas[, c(paste0("cv0_max_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
    dt1[, LL := NULL  ]
    
    ind <- which( bas$cv>=1 )
    pred <- bas[[ c(paste0("cv0_max_",fname,"_",feat1,"_",feat2)) ]][ind]
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2, AUC( bas$is_screener[ind] ,  pred ) ) )
  }else{
    dt1[, LL := fold1 ]
    dt <- dt1[, max(LL), keyby="patient_id" ]
    dt1[, LL := NULL  ]
    
    ind <- which( (bas$cv==1)&(bas$tr==0) )
    pred <- dt[J(bas$patient_id[ind])]$V1
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2 , AUC( bas$is_screener[ind] ,  pred ) ) )
  }
}


LIKELIHOOD_2WAY_CV_MEAN_MAX <- function( feat1="",feat2="", fname="", L=0, validate=F, calc_Mean_Max=T ){
  #   feat1 = "diagnosis_code"  
  #   feat2 = "primary_physician_role"  
  # fname=""  
  tic()
  if( fname=="" ){
    fname <- "_"
  }
  fold1 <- rep( NA , nrow(dt1) )
  fold2 <- rep( NA , nrow(dt1) )
  fold3 <- rep( NA , nrow(dt1) )
  foldA <- rep( NA , nrow(dt1) )
  
  p1 <- which( dt1$cv==1  )
  p2 <- which( dt1$cv==2  )
  p3 <- which( dt1$cv==3  )
  pt <- which( is.na(dt1$cv)  )
  
  DT <- data.table( var1=dt1[[feat1]] ,var2=dt1[[feat2]]  , tgt=dt1$tgt  )
  dt <- DT[,.N,by=c("var1","var2")]
  L <- median( dt$N  )
  rm(dt)
  if(L<4){
    L=4
  }
  print(paste("L",L));gc()
  
  if( validate==F ){
    ll123 <- LIKELIHOOD_2WAY( DT , L , c(p1,p2,p3)  )
    cat(',')
    ll1   <- LIKELIHOOD_2WAY( DT , L ,       p1 )
    cat(',')
    ll2   <- LIKELIHOOD_2WAY( DT , L ,       p2 )
    cat(',')
    ll3   <- LIKELIHOOD_2WAY( DT , L ,       p3 )
    cat(',')
    ll12  <- LIKELIHOOD_2WAY( DT , L , c(p1,p2) )
    cat(',')
    ll13  <- LIKELIHOOD_2WAY( DT , L , c(p1,p3) )
    cat(',')
    ll23  <- LIKELIHOOD_2WAY( DT , L , c(p2,p3) )
    cat('.')
  }else{
#     p2 <- p2[ sample.int(length(p2),0.5*length(p2))   ]
#     p1 <- p1[ sample.int(length(p1),0.5*length(p1))   ]
    ll23  <- LIKELIHOOD_2WAY( DT , L , p2 )
  }
  rm(DT)
  
  if( validate==F ){
    fold1[p1] <- ll23[p1]
    fold1[p2] <- ll3[p2]
    fold1[p3] <- ll2[p3]
    
    fold2[p2] <- ll13[p2]
    fold2[p1] <- ll3[p1]
    fold2[p3] <- ll1[p3]
    
    fold3[p3] <- ll12[p3]
    fold3[p1] <- ll2[p1]
    fold3[p2] <- ll1[p2]
    
    foldA[pt] <- ll123[pt]
    foldA[p1] <- ll23[p1]
    foldA[p2] <- ll13[p2]
    foldA[p3] <- ll12[p3]
    rm(ll1,ll2,ll3,ll12,ll13,ll123)
  }else{
    fold1[p1] <- ll23[p1]
  }
  
  if( validate==F ){
    cat(',')      
    #    dt1[, LL := NULL  ]
    dt1[, LL := fold1 ]
    if(calc_Mean_Max==T){
      dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
      bas[, c(paste0("cv1_mean_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv1_max_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
    }else{
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv1_max_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
    }
    
    cat(',')      
    dt1[, LL := NULL  ]
    dt1[, LL := fold2 ]
    if(calc_Mean_Max==T){
      dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
      bas[, c(paste0("cv2_mean_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv2_max_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
    }else{
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv2_max_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
    }
    
    cat(',')      
    dt1[, LL := NULL  ]
    dt1[, LL := fold3 ]
    if(calc_Mean_Max==T){
      dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
      bas[, c(paste0("cv3_mean_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv3_max_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
    }else{
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv3_max_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
    }
    
    cat(',')      
    dt1[, LL := NULL  ]
    dt1[, LL := foldA ]
    if(calc_Mean_Max==T){
      dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
      bas[, c(paste0("cv0_mean_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv0_max_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
    }else{
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv0_max_",fname,"_",feat1,"_",feat2)) := list(dt[J(bas$patient_id)]$V1)  ]
    }
    dt1[, LL := NULL  ]
    
    ind <- which( bas$cv>=1 )
    pred <- bas[[ c(paste0("cv0_mean_",fname,"_",feat1,"_",feat2)) ]][ind]
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2, AUC( bas$is_screener[ind] ,  pred ) ) )
    pred <- bas[[ c(paste0("cv0_max_",fname,"_",feat1,"_",feat2)) ]][ind]
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2, AUC( bas$is_screener[ind] ,  pred ) ) )
  }else{
    dt1[, LL := fold1 ]
    dt <- dt1[, max(LL), keyby="patient_id" ]
    dt2<- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
    dt1[, LL := NULL  ]
    
    ind <- which( (bas$cv==1)&(bas$tr==0) )
    pred <- dt[J(bas$patient_id[ind])]$V1
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2 , AUC( bas$is_screener[ind] ,  pred ) ) )
    pred <- dt2[J(bas$patient_id[ind])]$V1
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2 , AUC( bas$is_screener[ind] ,  pred ) ) )
  }
  toc()
}





LIKELIHOOD_3WAY_CV_MEAN_MAX <- function( feat1="",feat2="",feat3="", fname="", L=0, validate=F, calc_Mean_Max=T ){
  #   feat1 = "diagnosis_code"  
  #   feat2 = "primary_physician_role"  
  # fname=""  
  tic()
  if( fname=="" ){
    fname <- "_"
  }
  fold1 <- rep( NA , nrow(dt1) )
  fold2 <- rep( NA , nrow(dt1) )
  fold3 <- rep( NA , nrow(dt1) )
  foldA <- rep( NA , nrow(dt1) )
  
  p1 <- which( dt1$cv==1  )
  p2 <- which( dt1$cv==2  )
  p3 <- which( dt1$cv==3  )
  pt <- which( is.na(dt1$cv)  )
  
  DT <- data.table( var1=dt1[[feat1]], var2=dt1[[feat2]], var3=dt1[[feat3]]  , tgt=dt1$tgt  )
  dt <- DT[,.N,by=c("var1","var2","var3")]
  L <- median( dt$N  )
  rm(dt)
  if(L<3){
    L=3
  }
  print(paste("L",L));gc()
  
  if( validate==F ){
    ll123 <- LIKELIHOOD_3WAY( DT , L , c(p1,p2,p3)  )
    cat(',')
    ll1   <- LIKELIHOOD_3WAY( DT , L ,       p1 )
    cat(',')
    ll2   <- LIKELIHOOD_3WAY( DT , L ,       p2 )
    cat(',')
    ll3   <- LIKELIHOOD_3WAY( DT , L ,       p3 )
    cat(',')
    ll12  <- LIKELIHOOD_3WAY( DT , L , c(p1,p2) )
    cat(',')
    ll13  <- LIKELIHOOD_3WAY( DT , L , c(p1,p3) )
    cat(',')
    ll23  <- LIKELIHOOD_3WAY( DT , L , c(p2,p3) )
    cat('.')
  }else{
    #     p2 <- p2[ sample.int(length(p2),0.5*length(p2))   ]
    #     p1 <- p1[ sample.int(length(p1),0.5*length(p1))   ]
    ll23  <- LIKELIHOOD_3WAY( DT , L , p2 )
  }
  rm(DT)
  
  if( validate==F ){
    fold1[p1] <- ll23[p1]
    fold1[p2] <- ll3[p2]
    fold1[p3] <- ll2[p3]
    
    fold2[p2] <- ll13[p2]
    fold2[p1] <- ll3[p1]
    fold2[p3] <- ll1[p3]
    
    fold3[p3] <- ll12[p3]
    fold3[p1] <- ll2[p1]
    fold3[p2] <- ll1[p2]
    
    foldA[pt] <- ll123[pt]
    foldA[p1] <- ll23[p1]
    foldA[p2] <- ll13[p2]
    foldA[p3] <- ll12[p3]
    rm(ll1,ll2,ll3,ll12,ll13,ll123)
  }else{
    fold1[p1] <- ll23[p1]
  }
  
  if( validate==F ){
    cat(',')      
    #    dt1[, LL := NULL  ]
    dt1[, LL := fold1 ]
    if(calc_Mean_Max==T){
      dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
      bas[, c(paste0("cv1_mean_",fname,"_",feat1,"_",feat2,"_",feat3)) := list(dt[J(bas$patient_id)]$V1)  ]
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv1_max_",fname,"_",feat1,"_",feat2,"_",feat3)) := list(dt[J(bas$patient_id)]$V1)  ]
    }else{
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv1_max_",fname,"_",feat1,"_",feat2,"_",feat3)) := list(dt[J(bas$patient_id)]$V1)  ]
    }
    
    cat(',')      
    dt1[, LL := NULL  ]
    dt1[, LL := fold2 ]
    if(calc_Mean_Max==T){
      dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
      bas[, c(paste0("cv2_mean_",fname,"_",feat1,"_",feat2,"_",feat3)) := list(dt[J(bas$patient_id)]$V1)  ]
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv2_max_",fname,"_",feat1,"_",feat2,"_",feat3)) := list(dt[J(bas$patient_id)]$V1)  ]
    }else{
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv2_max_",fname,"_",feat1,"_",feat2,"_",feat3)) := list(dt[J(bas$patient_id)]$V1)  ]
    }
    
    cat(',')      
    dt1[, LL := NULL  ]
    dt1[, LL := fold3 ]
    if(calc_Mean_Max==T){
      dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
      bas[, c(paste0("cv3_mean_",fname,"_",feat1,"_",feat2,"_",feat3)) := list(dt[J(bas$patient_id)]$V1)  ]
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv3_max_",fname,"_",feat1,"_",feat2,"_",feat3)) := list(dt[J(bas$patient_id)]$V1)  ]
    }else{
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv3_max_",fname,"_",feat1,"_",feat2,"_",feat3)) := list(dt[J(bas$patient_id)]$V1)  ]
    }
    
    cat(',')      
    dt1[, LL := NULL  ]
    dt1[, LL := foldA ]
    if(calc_Mean_Max==T){
      dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
      bas[, c(paste0("cv0_mean_",fname,"_",feat1,"_",feat2,"_",feat3)) := list(dt[J(bas$patient_id)]$V1)  ]
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv0_max_",fname,"_",feat1,"_",feat2,"_",feat3)) := list(dt[J(bas$patient_id)]$V1)  ]
    }else{
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv0_max_",fname,"_",feat1,"_",feat2,"_",feat3)) := list(dt[J(bas$patient_id)]$V1)  ]
    }
    dt1[, LL := NULL  ]
    cat(',\n')      
    
    ind <- which( bas$cv>=1 )
    pred <- bas[[ c(paste0("cv0_mean_",fname,"_",feat1,"_",feat2,"_",feat3)) ]][ind]
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2, AUC( bas$is_screener[ind] ,  pred ) ) )
    pred <- bas[[ c(paste0("cv0_max_",fname,"_",feat1,"_",feat2,"_",feat3)) ]][ind]
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2, AUC( bas$is_screener[ind] ,  pred ) ) )
  }else{
    dt1[, LL := fold1 ]
    dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
    dt2<- dt1[, max(LL), keyby="patient_id" ]
    dt1[, LL := NULL  ]
    
    ind <- which( (bas$cv==1)&(bas$tr==0) )
    pred <- dt[J(bas$patient_id[ind])]$V1
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2 , AUC( bas$is_screener[ind] ,  pred ) ) )
    pred <- dt2[J(bas$patient_id[ind])]$V1
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2 , AUC( bas$is_screener[ind] ,  pred ) ) )
  }
  toc()
}



LIKELIHOOD_4WAY_CV_MEAN_MAX <- function( feat1="",feat2="",feat3="",feat4="", fname="", L=0, validate=F, calc_Mean_Max=T ){
  #   feat1 = "diagnosis_code"  
  #   feat2 = "primary_physician_role"  
  # fname=""  
  tic()
  if( fname=="" ){
    fname <- "_"
  }
  fold1 <- rep( NA , nrow(dt1) )
  fold2 <- rep( NA , nrow(dt1) )
  fold3 <- rep( NA , nrow(dt1) )
  foldA <- rep( NA , nrow(dt1) )
  
  p1 <- which( dt1$cv==1  )
  p2 <- which( dt1$cv==2  )
  p3 <- which( dt1$cv==3  )
  pt <- which( is.na(dt1$cv)  )
  
  DT <- data.table( var1=dt1[[feat1]], var2=dt1[[feat2]], var3=dt1[[feat3]], var4=dt1[[feat4]]  , tgt=dt1$tgt  )
  dt <- DT[,.N,by=c("var1","var2","var3","var4")]
  L <- median( dt$N  )
  rm(dt)
  if(L<3){
    L=3
  }
  print(paste("L",L));gc()
  
  if( validate==F ){
    ll123 <- LIKELIHOOD_4WAY( DT , L , c(p1,p2,p3)  )
    cat(',')
    ll1   <- LIKELIHOOD_4WAY( DT , L ,       p1 )
    cat(',')
    ll2   <- LIKELIHOOD_4WAY( DT , L ,       p2 )
    cat(',')
    ll3   <- LIKELIHOOD_4WAY( DT , L ,       p3 )
    cat(',')
    ll12  <- LIKELIHOOD_4WAY( DT , L , c(p1,p2) )
    cat(',')
    ll13  <- LIKELIHOOD_4WAY( DT , L , c(p1,p3) )
    cat(',')
    ll23  <- LIKELIHOOD_4WAY( DT , L , c(p2,p3) )
    cat('.')
  }else{
    ll23  <- LIKELIHOOD_4WAY( DT , L , p2 )
  }
  rm(DT)
  
  if( validate==F ){
    fold1[p1] <- ll23[p1]
    fold1[p2] <- ll3[p2]
    fold1[p3] <- ll2[p3]
    
    fold2[p2] <- ll13[p2]
    fold2[p1] <- ll3[p1]
    fold2[p3] <- ll1[p3]
    
    fold3[p3] <- ll12[p3]
    fold3[p1] <- ll2[p1]
    fold3[p2] <- ll1[p2]
    
    foldA[pt] <- ll123[pt]
    foldA[p1] <- ll23[p1]
    foldA[p2] <- ll13[p2]
    foldA[p3] <- ll12[p3]
    rm(ll1,ll2,ll3,ll12,ll13,ll123)
  }else{
    fold1[p1] <- ll23[p1]
  }
  
  if( validate==F ){
    cat(',')      
    dt1[, LL := fold1 ]
    if(calc_Mean_Max==T){
      dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
      bas[, c(paste0("cv1_mean_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) := list(dt[J(bas$patient_id)]$V1)  ]
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv1_max_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) := list(dt[J(bas$patient_id)]$V1)  ]
    }else{
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv1_max_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) := list(dt[J(bas$patient_id)]$V1)  ]
    }
    
    cat(',')      
    dt1[, LL := NULL  ]
    dt1[, LL := fold2 ]
    if(calc_Mean_Max==T){
      dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
      bas[, c(paste0("cv2_mean_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) := list(dt[J(bas$patient_id)]$V1)  ]
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv2_max_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) := list(dt[J(bas$patient_id)]$V1)  ]
    }else{
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv2_max_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) := list(dt[J(bas$patient_id)]$V1)  ]
    }
    
    cat(',')      
    dt1[, LL := NULL  ]
    dt1[, LL := fold3 ]
    if(calc_Mean_Max==T){
      dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
      bas[, c(paste0("cv3_mean_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) := list(dt[J(bas$patient_id)]$V1)  ]
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv3_max_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) := list(dt[J(bas$patient_id)]$V1)  ]
    }else{
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv3_max_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) := list(dt[J(bas$patient_id)]$V1)  ]
    }
    
    cat(',')      
    dt1[, LL := NULL  ]
    dt1[, LL := foldA ]
    if(calc_Mean_Max==T){
      dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
      bas[, c(paste0("cv0_mean_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) := list(dt[J(bas$patient_id)]$V1)  ]
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv0_max_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) := list(dt[J(bas$patient_id)]$V1)  ]
    }else{
      dt <- dt1[, max(LL), keyby="patient_id" ]
      bas[, c(paste0("cv0_max_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) := list(dt[J(bas$patient_id)]$V1)  ]
    }
    dt1[, LL := NULL  ]
    cat(',\n')      
    
    ind <- which( bas$cv>=1 )
    pred <- bas[[ c(paste0("cv0_mean_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) ]][ind]
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2, feat3, feat4, AUC( bas$is_screener[ind] ,  pred ) ) )
    pred <- bas[[ c(paste0("cv0_max_",fname,"_",feat1,"_",feat2,"_",feat3,"_",feat4)) ]][ind]
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2, feat3, feat4, AUC( bas$is_screener[ind] ,  pred ) ) )
  }else{
    dt1[, LL := fold1 ]
    dt <- dt1[, mean(LL,na.rm=T), keyby="patient_id" ]
    dt2<- dt1[, max(LL), keyby="patient_id" ]
    dt1[, LL := NULL  ]
    
    ind <- which( (bas$cv==1)&(bas$tr==0) )
    pred <- dt[J(bas$patient_id[ind])]$V1
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2, feat3, feat4 , AUC( bas$is_screener[ind] ,  pred ) ) )
    pred <- dt2[J(bas$patient_id[ind])]$V1
    pred[pred == Inf] <- NA
    pred[pred == -Inf] <- NA
    pred[is.na(pred)] <- mean(pred, na.rm=T)
    print( paste( feat1, feat2, feat3, feat4 , AUC( bas$is_screener[ind] ,  pred ) ) )
  }
  toc()
}


