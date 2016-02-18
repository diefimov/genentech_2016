rm(list=ls())
source("fn.base.r");gc()

# ##################################################
train = fread( '../data/input/patients_train.csv'  )
test  = fread( '../data/input/patients_test.csv'  );gc()

train[, tr:=0]
test[ , tr:=1]
test[ , is_screener:=NA]

bas <- rbind( train,test );
bas[, patient_gender := NULL ]
rm(train,test);gc()

dt <- fread( '../data/input/train_cv.csv'  );gc()
setkeyv( dt , "patient_id" )
bas[, cv := dt[J(bas$patient_id)]$cv_index  ]
table(bas$cv)

setkeyv( bas, "patient_id"  )
fn.save.data("bas")
rm(raw,bas, train, test, dt);gc()
##############################################################


##############################################################
fn.load.data("bas")

dt1 <- fread('../data/input/patient_activity_head.csv', header=T );gc()
setkeyv( dt1, c("patient_id","activity_year","activity_month")  );gc()
dt1[, date := (activity_year-2008)*12 + activity_month ];gc()

ind <- which( dt1$patient_id %in% bas$patient_id );gc()
dt1 <- dt1[ ind ];gc()
dt1[, cv := bas[J(dt1$patient_id)]$cv ];gc()

dt <- dt1[, sum(activity_type=="R",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f1_0 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(activity_type=="A",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f1_1 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, 96-max(date) , keyby=c("patient_id") ];gc()
bas[, f1_2 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, (max(date)-min(date))/(1+.N) , keyby=c("patient_id") ];gc()
bas[, f1_3 := dt[J(bas$patient_id)]$V1 ]

dt1[, y := floor(date/12)]
dt <- dt1[, .N , keyby=c("patient_id","y") ];gc()
dt <- dt[, max(N) , keyby=c("patient_id") ];gc()
bas[, f1_4 := dt[J(bas$patient_id)]$V1 ];gc()

dt <- dt1[, max(date)-min(date) , keyby=c("patient_id") ];gc()
bas[, f1_5 := dt[J(bas$patient_id)]$V1 ];gc()

dt <- dt1[, 96-min(date) , keyby=c("patient_id") ];gc()
bas[, f1_6 := dt[J(bas$patient_id)]$V1 ]

dt <- dt1[, .N , keyby=c("patient_id") ];gc()
bas[, f1_N := dt[J(bas$patient_id)]$N ]

dt1[, N_by_year := .N , by=c("patient_id","activity_year") ]
dt <- dt1[, mean(N_by_year) , keyby=c("patient_id") ];gc()
bas[, f1_7 := dt[J(bas$patient_id)]$V1 ]

dt1[, N_by_month := .N , by=c("patient_id","date") ]
dt <- dt1[, mean(N_by_month) , keyby=c("patient_id") ];gc()
bas[, f1_8 := dt[J(bas$patient_id)]$V1 ]

rm(dt, dt1);gc()
##############################################################################33


##############################################################################33
dt1 <- fread('../data/input/surgical_head.csv', header=T );gc()
setkeyv( dt1, c("patient_id","surgical_procedure_date")  );gc()

dt2 <- fread('../data/input/surgical_code.csv', header=T );gc()
dt2$nc <- nchar(dt2$surgical_description)
dt2$fw <- as.numeric( factor( substr(dt2$surgical_description,1,4) ) )

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

dt1[, claim_type:=NULL]

dt <- dt1[, .N , keyby=c("patient_id") ];gc()
bas[, f2_N := dt[J(bas$patient_id)]$N ]
dt <- dt1[, sum(procedure_type_code=="HXPR",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_0 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(procedure_type_code=="HX05",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_1 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(procedure_type_code=="HX01",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_2 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(procedure_type_code=="HX04",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_3 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(procedure_type_code=="HX02",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_4 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(procedure_type_code=="HX03",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_5 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(procedure_type_code=="0001",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_6 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(procedure_type_code=="0002",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_7 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(procedure_type_code=="0003",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_8 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(procedure_type_code=="0004",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_9 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(procedure_type_code=="0005",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_10 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(procedure_type_code=="0006",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_11 := dt[J(bas$patient_id)]$V1 ]

dt <- dt1[, sum(place_of_service=="INPATIENT",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_12 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(place_of_service=="OUTPATIENT",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_13 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(place_of_service=="OTHER",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_14 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(place_of_service=="CLINIC",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_15 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(place_of_service=="UNKNOWN",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_16 := dt[J(bas$patient_id)]$V1 ]

dt <- dt1[, sum(plan_type=="COMMERCIAL",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_17 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(plan_type=="MEDICARE",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_18 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(plan_type=="MEDICAID",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_19 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(plan_type=="GOVERNMENT",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_20 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(plan_type=="CASH",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_21 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(plan_type=="UNKNOWN",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_22 := dt[J(bas$patient_id)]$V1 ]

dt <- dt1[, sum(primary_physician_role=="ATG",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_23:= dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(primary_physician_role=="",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_24:= dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(primary_physician_role=="OTH",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_25:= dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum(primary_physician_role=="OPR",na.rm=T) , keyby=c("patient_id") ];gc()
bas[, f2_26:= dt[J(bas$patient_id)]$V1 ]

dt1[, N1 := .N , keyby=c("procedure_type_code","place_of_service") ];gc()
dt <- dt1[, mean(N1) , keyby=c("patient_id") ];gc()
bas[, f2_N2 := dt[J(bas$patient_id)]$V1 ]

dt1[, N1 := .N , keyby=c("procedure_type_code","place_of_service","plan_type") ];gc()
dt <- dt1[, mean(N1) , keyby=c("patient_id") ];gc()
bas[, f2_N3 := dt[J(bas$patient_id)]$V1 ]

dt1[, N1 := .N , keyby=c("procedure_type_code","place_of_service","plan_type","primary_physician_role") ];gc()
dt <- dt1[, mean(N1) , keyby=c("patient_id") ];gc()
bas[, f2_N4 := dt[J(bas$patient_id)]$V1 ]

rm(dt1,dt2,dt);gc()
########################################################################

dim(bas)
build1 <- copy(bas)
fn.save.data("build1")
rm(build1);gc()
########################################################################

