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
dt1[, claim_type:=NULL ]
dt1[, primary_practitioner_id:=NULL ]
dt1[, primary_physician_role:=NULL ];gc()
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


tmp1 <- dt1[ claim_id==64857014327148  ]
tmp2 <- dt2[ claim_id==64857014327148  ];gc()

d1 <- dt1[, .N, by="claim_id"]
d2 <- dt2[, .N, by="claim_id"];gc()


d1 <- dt1[, .N, keyby="diagnosis_date"];gc()
d1$N <- as.numeric( substr( d1$diagnosis_date,1,4 ) )
dt1[, year := d1[J(dt1$diagnosis_date)]$N  ];gc()
dt1[, diagnosis_date:=NULL ];gc()

d1 <- dt1[, .N, keyby="diagnosis_code"];gc()
d1 <- d1[ order(N,decreasing=T) ]
ind <- which( d1$N==1  )
d1$newid <- 1:nrow(d1)
d1$newid[ ind  ] <- 16243
setkeyv( d1, "diagnosis_code"  )
tmp <- d1[J( dt1$diagnosis_code  )]$newid ; gc()
dt1[, diagnosis_code:=NULL ];gc()
dt1[, diagnosis_code:=tmp ];gc()

###
d1 <- dt2[, .N, keyby="procedure_date"];gc()
d1$N <- as.numeric( substr( d1$procedure_date,1,4 ) )
setkeyv( d1 , "procedure_date"  )
dt2[, year := d1[J(dt2$procedure_date)]$N  ];gc()
dt2[, procedure_date:=NULL ];gc()

d1 <- dt2[, .N, keyby="procedure_code"];gc()
d1 <- d1[ order(N,decreasing=T) ]
ind <- which( d1$N==1  )
d1$newid <- 1:nrow(d1)
d1$newid[ ind  ] <- 15289
setkeyv( d1, "procedure_code"  )
tmp <- d1[J( dt2$procedure_code  )]$newid ; gc()
dt2[, procedure_code:=NULL ];gc()
dt2[, procedure_code:=tmp ];gc()
rm(tmp,d1,d2,dt,tmp1,tmp2);gc()
###

dt1[, flag1 := .N , by="claim_id" ];gc()

dt <- dt2[, .N , keyby="claim_id"];gc()
dt1[, flag2 := dt[J(dt1$claim_id)]$N ];gc()
dt <- dt2[, .N , keyby="claim_id"];gc()

dt <- dt2[, max(claim_line_item)-.N , keyby="claim_id"];gc()
dt1[, flag3 := dt[J(dt1$claim_id)]$V1 ];gc()

LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="flag3", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "diagnosis_code flag3 0.70015581008066"
# [1] "diagnosis_code flag3 0.869720991472213"

dt1[, flag4 := 1*flag1+8*flag2+64*flag3 ];gc()
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="flag4", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "diagnosis_code flag4 0.699704580841202"
# [1] "diagnosis_code flag4 0.865480501537263"

LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="flag2", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# "diagnosis_code flag2 0.698352605625026"
# "diagnosis_code flag2 0.8486243207131"

LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="flag1", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "diagnosis_code flag1 0.691639007229912"
# [1] "diagnosis_code flag1 0.819800551188795"

LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="flag1",feat2="flag2", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "flag1 flag2 0.626513368572039"
# [1] "flag1 flag2 0.736962364637853"

LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="flag1",feat2="flag3", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "flag1 flag3 0.624829440645681"
# [1] "flag1 flag3 0.741177380576334"

LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="flag2",feat2="flag3", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "flag2 flag3 0.523400039276578"
# [1] "flag2 flag3 0.515944214935269"

LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="year",feat2="flag1", fname="", L=0, validate=F, calc_Mean_Max=T );gc()
# [1] "year flag1 0.574422837729412"
# [1] "year flag1 0.490156358806583"

build4 <- copy(bas)
fn.save.data("build4")
rm(build4);gc()
########################################################################
