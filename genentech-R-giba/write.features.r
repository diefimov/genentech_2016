rm(list=ls())
source("fn.base.r");gc()

fn.load.data("build2")
fn.load.data("build3")
fn.load.data("build4")
fn.load.data("build5")
dim(build2)
dim(build3)
dim(build4)
dim(build5)

build4[, patient_id := NULL ]
build4[, patient_age_group := NULL ]
build4[, patient_state := NULL ]
build4[, ethinicity := NULL ]
build4[, household_income := NULL ]
build4[, education_level := NULL ]
build4[, is_screener := NULL ]
build4[, tr := NULL ]
build4[, cv := NULL ];gc()
colnames(build3)

build3[, patient_id := NULL ]
build3[, patient_age_group := NULL ]
build3[, patient_state := NULL ]
build3[, ethinicity := NULL ]
build3[, household_income := NULL ]
build3[, education_level := NULL ]
build3[, is_screener := NULL ]
build3[, tr := NULL ]
build3[, cv := NULL ];gc()
colnames(build3)

build2[, patient_id := NULL ]
build2[, patient_age_group := NULL ]
build2[, patient_state := NULL ]
build2[, ethinicity := NULL ]
build2[, household_income := NULL ]
build2[, education_level := NULL ]
build2[, is_screener := NULL ]
build2[, tr := NULL ]
build2[, cv := NULL ];gc()
colnames(build2)

bas <- cbind( build5, build4, build3, build2 )
rm( build5, build2, build3, build4, dt );gc()

BAS <- copy( bas )
for (f in colnames(BAS) ) {
  if ( class(BAS[[f]])=="character" | class(BAS[[f]])=="Date" ) {
    print(f)
    BAS[[f]] <- as.numeric(factor(BAS[[f]]))
  }
}
for (f in colnames(BAS) ) {
  print(f)
  tmp <- as.numeric( BAS[[f]] )
  tmp <- round( tmp , digits=6 )
  tmp[ tmp>9e9 ] <- NA
  tmp[ tmp< -9e9 ] <- NA
  tmp[ is.na(tmp) ] <- -999
  BAS[[f]] <- tmp
  
};gc()
colnames(BAS)

BAS[, patient_age_group := NULL ]
BAS[, patient_state := NULL ]
BAS[, ethinicity := NULL ]
BAS[, household_income := NULL ]
BAS[, education_level := NULL ]

cols <- colnames(BAS)
feats.base = cols[ which( substr(cols,1,2)!="cv"  ) ]
feats.cv0 = cols[ which( substr(cols,1,3)=="cv0"  ) ]
feats.cv1 = cols[ which( substr(cols,1,3)=="cv1"  ) ]
feats.cv2 = cols[ which( substr(cols,1,3)=="cv2"  ) ]
feats.cv3 = cols[ which( substr(cols,1,3)=="cv3"  ) ]

BAS0 <- BAS[, c(feats.base,feats.cv0) , with=F  ]
BAS1 <- BAS[, c(feats.base,feats.cv1) , with=F  ]
BAS2 <- BAS[, c(feats.base,feats.cv2) , with=F  ]
BAS3 <- BAS[, c(feats.base,feats.cv3) , with=F  ]
BAS1 <- BAS1[tr==0]
BAS2 <- BAS2[tr==0]
BAS3 <- BAS3[tr==0]
dim(BAS0)
dim(BAS1)
dim(BAS2)
dim(BAS3)

BAS0[ , tr := NULL ]
BAS1[ , tr := NULL ]
BAS2[ , tr := NULL ]
BAS3[ , tr := NULL ]
BAS0[ , is_screener := NULL ]
BAS1[ , is_screener := NULL ]
BAS2[ , is_screener := NULL ]
BAS3[ , is_screener := NULL ]

setkeyv( BAS0, "patient_id" )
setkeyv( BAS1, "patient_id" )
setkeyv( BAS2, "patient_id" )
setkeyv( BAS3, "patient_id" );gc()

BAS0 <- as.matrix( BAS0 )
BAS1 <- as.matrix( BAS1 )
BAS2 <- as.matrix( BAS2 )
BAS3 <- as.matrix( BAS3 );gc()

# merge Vowpal Wabbit feature
fn.load.data("vw1")
str(vw1)
setkeyv( vw1, "patient_id"  )
ind <- which( BAS$tr==0 )
BAS0 <- cbind( BAS0 , vw1=vw1$predict_screener  )
vw1 <- vw1[ tr==0 ]
BAS1 <- cbind( BAS1 , vw1=vw1$predict_screener  )
BAS2 <- cbind( BAS2 , vw1=vw1$predict_screener  )
BAS3 <- cbind( BAS3 , vw1=vw1$predict_screener  );gc()


print("WRITING FOLD 1...")
fp <- gzfile("../data/team/giba/train.giba.cv1.csv.gz", "w")
write.csv(BAS1, fp  , row.names = F, quote=F )
close(fp);gc()

print("WRITING FOLD 2...")
fp <- gzfile("../data/team/giba/train.giba.cv2.csv.gz", "w")
write.csv(BAS2, fp  , row.names = F, quote=F )
close(fp);gc()

print("WRITING FOLD 3...")
fp <- gzfile("../data/team/giba/train.giba.cv3.csv.gz", "w")
write.csv(BAS3, fp  , row.names = F, quote=F )
close(fp);gc()

print("WRITING FOLD 4...")
fp <- gzfile("../data/team/giba/train.giba.cv0.csv.gz", "w")
write.csv(BAS0, fp  , row.names = F, quote=F )
close(fp);gc()

