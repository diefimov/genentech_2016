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

##############################################################################33
dt1 <- fread('../data/input/diagnosis_head.csv', header=T );gc()
setkeyv( dt1, c("patient_id","claim_id","diagnosis_date")  );gc()

del.train = fread( '../data/input/train_patients_to_exclude.csv', header=F  ) 
del.test  = fread( '../data/input/test_patients_to_exclude.csv', header=F  ) 

ind <- which( !dt1$patient_id %in% c( del.train$V1 , del.test$V1  ) );gc()
dt1 <- dt1[ ind ];gc()

dt2 <- fread('../data/input/procedure_head.csv', header=T );gc()
dt2[, operating_practitioner_id := NULL ];gc()
dt2[, ordering_practitioner_id := NULL ];gc()
dt2[, rendering_practitioner_id := NULL ];gc()
dt2[, referring_practitioner_id := NULL ];gc()
dt2[, attending_practitioner_id := NULL ];gc()
dt2[, claim_type := NULL ];gc()
dt2[, place_of_service := NULL ];gc()
dt2[, plan_type := NULL ];gc()
dt2[, primary_physician_role := NULL ];gc()

ind <- which( !dt2$patient_id %in% c( del.train$V1 , del.test$V1  ) );gc()
dt2 <- dt2[ ind ];gc()

setkeyv( dt2, c("patient_id","procedure_date","claim_id","claim_line_item") );gc()

bas1 <- copy(bas[, colnames(bas)[c(1,7,9)], with=F ]);gc()
setkeyv( bas1, "patient_id"  )
dt1[, cv := bas1[J(dt1$patient_id)]$cv ];gc()
dt1[, tgt:= bas1[J(dt1$patient_id)]$is_screener ];gc()
dt2[, tgt:= bas1[J(dt1$patient_id)]$is_screener ];gc()

dt1[, flag1 := .N , by="claim_id" ];gc()

dt <- dt2[, .N , keyby="claim_id"];gc()
dt1[, flag2 := dt[J(dt1$claim_id)]$N ];gc()
dt <- dt2[, .N , keyby="claim_id"];gc()

dt <- dt2[, max(claim_line_item)-.N , keyby="claim_id"];gc()
dt1[, flag3 := dt[J(dt1$claim_id)]$V1 ];gc()

LIKELIHOOD_3WAY_CV_MEAN_MAX( feat1="flag1",feat2="flag2",feat3="flag3", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "flag1 flag2 0.628495839033553"
# [1] "flag1 flag2 0.746795549980462"

LIKELIHOOD_3WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="flag1",feat3="flag2", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "diagnosis_code flag1 0.697171946454153"
# [1] "diagnosis_code flag1 0.84842835885366"

LIKELIHOOD_3WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="flag1",feat3="flag3", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "diagnosis_code flag1 0.698245598512909"
# [1] "diagnosis_code flag1 0.871069252697617"

LIKELIHOOD_3WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="flag2",feat3="flag3", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "diagnosis_code flag2 0.701579771271569"
# [1] "diagnosis_code flag2 0.868503268308118"

LIKELIHOOD_3WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="flag2",feat3="flag3", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "diagnosis_code flag2 0.701579771271569"
# [1] "diagnosis_code flag2 0.868503268308118"

LIKELIHOOD_3WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="flag2",feat3="flag3", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "diagnosis_code flag2 0.701579771271569"
# [1] "diagnosis_code flag2 0.868503268308118"

LIKELIHOOD_4WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="primary_practitioner_id",feat3="flag3",feat4="claim_type", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "diagnosis_code primary_practitioner_id flag3 claim_type 0.733064070501767"
# [1] "diagnosis_code primary_practitioner_id flag3 claim_type 0.812059332764248"

LIKELIHOOD_4WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="primary_practitioner_id",feat3="flag3",feat4="primary_physician_role", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "diagnosis_code primary_practitioner_id flag3 primary_physician_role 0.732748793136208"
# [1] "diagnosis_code primary_practitioner_id flag3 primary_physician_role 0.810763625420145"

LIKELIHOOD_4WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="primary_practitioner_id",feat3="flag2",feat4="claim_type", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "diagnosis_code primary_practitioner_id flag2 claim_type 0.733526674227745"
# [1] "diagnosis_code primary_practitioner_id flag2 claim_type 0.80776922391907"

LIKELIHOOD_4WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="primary_practitioner_id",feat3="flag2",feat4="primary_physician_role", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "diagnosis_code primary_practitioner_id flag2 primary_physician_role 0.73302958928401"
# [1] "diagnosis_code primary_practitioner_id flag2 primary_physician_role 0.80686953602814"

rm(dt1,dt2,dt);gc()

build5 <- copy(bas)
fn.save.data("build5")
rm(build5);gc()
########################################################################
