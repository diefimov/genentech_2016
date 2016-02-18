rm(list=ls())
source("fn.base.r");gc()

fn.load.data("build1")
bas <- copy(build1)
rm(build1);gc()

bas <- bas[, colnames(bas)[1:9], with=F  ];gc()

i=3
feat=0
for( i in 2:6  ){
  print(i)
  cn <- c(colnames(bas)[i])
  dt <- bas[, .N , by=cn  ]
  dt$N <- feat + c(1:nrow(dt))
  setkeyv(dt,cn)
  bas[, c(cn) := list(dt[J(bas[,cn,with=F])]$N) ] 
  feat = max(bas[[cn]])
}
gc()


##############################################################################33
dt1 <- fread('../data/input/surgical_head.csv', header=T );gc()
setkeyv( dt1, c("patient_id","surgical_procedure_date")  );gc()

dt2 <- fread('../data/input/surgical_code.csv', header=T );gc()
dt2$nc <- nchar(dt2$surgical_description)
dt2$fw <- as.numeric( factor( substr(dt2$surgical_description,1,4) ) )
table(dt2$fw)

lw <- dt2$surgical_description
for( i in 1:nrow(dt2)){
  lw[i] <- substr( lw[i] , dt2$nc[i]-3, dt2$nc[i]  )  
}
dt2$lw <- as.numeric( factor( lw ) )

dt2 <- dt2[ !duplicated(dt2$surgical_code) ]
setkeyv( dt2, "surgical_code"  );gc()
dt1[, sc1 := dt2[J(dt1$surgical_code)]$nc ]
dt1[, sc2 := dt2[J(dt1$surgical_code)]$fw ]
dt1[, sc3 := dt2[J(dt1$surgical_code)]$lw ]


dt2 <- fread('../data/input/physicians.csv', header=T );gc()
setkeyv( dt2 , "practitioner_id"  )

dt1[, pid_1 := dt2[J(dt1$practitioner_id)]$physician_id ]
dt1[, pid_2 := dt2[J(dt1$practitioner_id)]$state ]
dt1[, pid_3 := dt2[J(dt1$practitioner_id)]$specialty_code ]
dt1[, pid_4 := dt2[J(dt1$practitioner_id)]$specialty_description ]
dt1[, pid_5 := dt2[J(dt1$practitioner_id)]$CBSA ]
rm(dt2);gc()


i=3
feat=0
for( i in 3:ncol(dt1)  ){
  print(i)
  cn <- c(colnames(dt1)[i])
  dt <- dt1[, .N , by=cn  ]
  dt$N <- feat + c(1:nrow(dt))
  setkeyv(dt,cn)
  dt1[, c(cn) := list(dt[J(dt1[,cn,with=F])]$N) ] 
  feat = max(dt1[[cn]])
}


dt <- dt1[, list(
  paste(procedure_type_code,collapse=" " ),
  paste(claim_type,collapse=" " ),
  paste(surgical_code,collapse=" " ),
  paste(surgical_procedure_date,collapse=" " ),
  paste(place_of_service,collapse=" " ),
  paste(plan_type,collapse=" " ),
  paste(practitioner_id,collapse=" " ),
  paste(primary_physician_role,collapse=" " ),
  paste(sc1,collapse=" " ),
  paste(sc2,collapse=" " ),
  paste(sc3,collapse=" " ),
  paste(pid_1,collapse=" " ),
  paste(pid_2,collapse=" " ),
  paste(pid_3,collapse=" " ),
  paste(pid_4,collapse=" " ),
  paste(pid_5,collapse=" " )
), keyby="patient_id" ]



bas[, sh1:=dt[J(bas$patient_id)]$V1  ]
bas[, sh2:=dt[J(bas$patient_id)]$V2  ]
bas[, sh3:=dt[J(bas$patient_id)]$V3  ]
bas[, sh4:=dt[J(bas$patient_id)]$V4  ]
bas[, sh5:=dt[J(bas$patient_id)]$V5  ]
bas[, sh6:=dt[J(bas$patient_id)]$V6  ]
bas[, sh7:=dt[J(bas$patient_id)]$V7  ]
bas[, sh8:=dt[J(bas$patient_id)]$V8  ]
bas[, sh9:=dt[J(bas$patient_id)]$V9  ]
bas[, sh10:=dt[J(bas$patient_id)]$V10  ]
bas[, sh11:=dt[J(bas$patient_id)]$V11  ]
bas[, sh12:=dt[J(bas$patient_id)]$V12  ]
bas[, sh13:=dt[J(bas$patient_id)]$V13  ]
bas[, sh14:=dt[J(bas$patient_id)]$V14  ]


bas[, patient_age_group := paste0( "|a ",patient_age_group, " " )  ]
bas[, patient_state := paste0( patient_state, " " )  ]
bas[, ethinicity := paste0( ethinicity, " " )  ]
bas[, household_income := paste0( household_income, " " )  ]
bas[, education_level := paste0(  education_level, " " )  ];gc()
bas[, sh1 := paste0( "|b ", sh1, " " )  ]
bas[, sh2 := paste0( "|c ", sh2, " " )  ]
bas[, sh3 := paste0( "|d ", sh3, " " )  ]
bas[, sh4 := paste0( "|e ", sh4, " " )  ]
bas[, sh5 := paste0( "|f ", sh5, " " )  ]
bas[, sh6 := paste0( "|g ", sh6, " " )  ]
bas[, sh7 := paste0( "|h ", sh7, " " )  ]
bas[, sh8 := paste0( "|i ", sh8, " " )  ]
bas[, sh9 := paste0( "|j ", sh9, " " )  ]
bas[, sh10 := paste0( "|k ", sh10, " " )  ]
bas[, sh11 := paste0( "|l ", sh11, " " )  ]
bas[, sh12 := paste0( "|m ", sh12, " " )  ]
bas[, sh13 := paste0( "|n ", sh13, " " )  ]
bas[, sh14 := paste0( "|o ", sh14, " " )  ]


cols <- colnames(bas)
cols <- cols[c(-1,-7,-8,-9)]
cols <- c( "patient_id", "tr" , "cv" , "is_screener" , cols  )
bas <- bas[, cols , with=F]
bas[, is_screener := paste0( is_screener, " " )  ]
bas[, is_screener := ifelse( is_screener=="1 ", "1 ","-1 "  )  ]
target <- ifelse( bas$is_screener[bas$tr==0]=="1 " , 1 , 0 )

predtrain <- rep( 0 , sum( bas$tr==0 ) ) 
for( fold in 1:3  ){
  tr <- bas[ tr==0 ]
  py <- which( tr$cv==fold   )
  val <- tr[ cv==fold  ]
  tr  <- tr[ cv!=fold  ]

  tr[, patient_id := NULL  ]
  tr[, tr := NULL  ]
  tr[, cv := NULL  ]
  val[, patient_id := NULL  ]
  val[, tr := NULL  ]
  val[, cv := NULL  ]
  tgt <- ifelse( val$is_screener=="1 " , 1 , 0 )
  
  write.table( val , 'val.vw', row.names=F, col.names=F, quote=F, sep=""  )
  for( bag in 1:5){
    print(bag)
    shf <- sample.int( nrow(tr)  )
    write.table( tr[shf,] , 'train.vw', row.names=F, col.names=F, quote=F, sep=""  )
    system( 'vw --data train.vw -f model.vw -l 0.03 -b 27 --loss_function logistic -P 100000 --holdout_off --ignore j --ignore k -qaa -c -k --passes 3 --nn 8 ' )
    system( 'vw val.vw -t -i model.vw -p pred.vw -P 100000 --loss_function logistic ' )
    pred <- fread( 'pred.vw', header=F )
    predtrain[py] <- predtrain[py] + pred$V1
    print( AUC( tgt , predtrain[py] )  )  
  }
}
print( AUC(  target , predtrain ) )

tr <- bas[ tr==0 ]
ts <- bas[ tr==1 ]
tr[, patient_id := NULL  ]
tr[, tr := NULL  ]
tr[, cv := NULL  ]
ts[, patient_id := NULL  ]
ts[, tr := NULL  ]
ts[, cv := NULL  ]
write.table( ts , 'test.vw', row.names=F, col.names=F, quote=F, sep=""  )
predtest <- rep( 0 , nrow(ts) ) 
for( bag in 1:5){
  print(bag)
  shf <- sample.int( nrow(tr)  )
  write.table( tr[shf,] , 'train.vw', row.names=F, col.names=F, quote=F, sep=""  )
  system( 'vw --data train.vw -f model.vw -l 0.03 -b 27 --loss_function logistic -P 100000 --holdout_off --ignore j --ignore k -qaa -c -k --passes 3 --nn 8 ' )
  system( 'vw test.vw -t -i model.vw -p pred.vw -P 100000 --loss_function logistic ' )
  pred <- fread( 'pred.vw', header=F )
  predtest <- predtest + pred$V1
}
tr <- bas[ tr==0 ]
ts <- bas[ tr==1 ]
vw1 <- data.table(
  patient_id = c( tr$patient_id , ts$patient_id ),
  predict_screener = c( predtrain , predtest ),
  is_screener = ifelse( c( tr$is_screener , rep( NA, nrow(ts)  ) )=="1 ", 1 , 0)  ,
  cv = c( tr$cv , ts$cv ),
  tr = c( tr$tr , ts$tr )
)
fn.save.data("vw1")

ind <- which( vw1$tr==0 & vw1$cv>=1 )
print( AUC( vw1$is_screener[ind] , vw1$predict_screener[ind] )  )

rm(dt1,dt,bas,pred,tr,ts,val,vw1);gc()
