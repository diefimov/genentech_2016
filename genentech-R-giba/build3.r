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


dt1 <- fread('../data/input/diagnosis_head.csv', header=T );gc()
setkeyv( dt1, c("patient_id","diagnosis_date")  );gc()

ind <- which( dt1$patient_id %in% bas$patient_id );gc()
dt1 <- dt1[ ind ];gc()

bas1 <- copy(bas[, colnames(bas)[c(1,7,9)], with=F ]);gc()
setkeyv( bas1, "patient_id"  )
dt1[, cv := bas1[J(dt1$patient_id)]$cv ];gc()
dt1[, tgt:= bas1[J(dt1$patient_id)]$is_screener ];gc()

dt1[, claim_id:=NULL ];gc()

dt2 <- fread('../data/input/physicians.csv', header=T );gc()
setkeyv( dt2 , "practitioner_id"  )

dt1[, pid_dh1 := dt2[J(dt1$primary_practitioner_id)]$physician_id ];gc()
dt1[, pid_dh2 := dt2[J(dt1$primary_practitioner_id)]$state ];gc()
dt1[, pid_dh3 := dt2[J(dt1$primary_practitioner_id)]$specialty_code ];gc()
dt1[, pid_dh4 := dt2[J(dt1$primary_practitioner_id)]$CBSA ];gc()
rm(dt2);gc()


LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="claim_type", fname="", L=0, validate=F, calc_Mean_Max=T )
# "diagnosis_code claim_type 0.739321912536681"
# "diagnosis_code claim_type 0.856515363208506"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="diagnosis_date", fname="dh", L=0, validate=F , calc_Mean_Max=T)
# "diagnosis_code diagnosis_date 0.737810067758258"
# "diagnosis_code diagnosis_date 0.847854306797832"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="primary_practitioner_id", fname="dh", L=0, validate=F , calc_Mean_Max=T)
# "diagnosis_code primary_practitioner_id 0.775937386353969"
# "diagnosis_code primary_practitioner_id 0.837442268499513"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="primary_physician_role", fname="dh", L=0, validate=F , calc_Mean_Max=T)
# "diagnosis_code primary_physician_role 0.740080359917819"
# "diagnosis_code primary_physician_role 0.870296126366153"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="pid_dh1", fname="dh", L=0, validate=F , calc_Mean_Max=T)
# "diagnosis_code pid_dh1 0.775775065506869"
# "diagnosis_code pid_dh1 0.837232986201449"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="pid_dh2", fname="dh", L=0, validate=F , calc_Mean_Max=T)
# "diagnosis_code pid_dh2 0.731555026752539"
# "diagnosis_code pid_dh2 0.856026205184023"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="pid_dh3", fname="dh", L=0, validate=F , calc_Mean_Max=T)
# "diagnosis_code pid_dh3 0.748757424742263"
# "diagnosis_code pid_dh3 0.867634605636215"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="diagnosis_code",feat2="pid_dh4", fname="dh", L=0, validate=F , calc_Mean_Max=T)
#"diagnosis_code pid_dh4 0.744150938287505"
#"diagnosis_code pid_dh4 0.846792438053783"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="primary_physician_role",feat2="pid_dh1", fname="dh", L=0, validate=F , calc_Mean_Max=T)
# "primary_physician_role pid_dh1 0.7521004857359"
# "primary_physician_role pid_dh1 0.769446690749941"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="primary_physician_role",feat2="pid_dh3", fname="dh", L=0, validate=F , calc_Mean_Max=T)
# "primary_physician_role pid_dh3 0.712941476645959"
# "primary_physician_role pid_dh3 0.710644741332741"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="primary_physician_role",feat2="pid_dh4", fname="dh", L=0, validate=F , calc_Mean_Max=T)
# "primary_physician_role pid_dh4 0.639265391360297"
# "primary_physician_role pid_dh4 0.594366124721873"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="primary_physician_role",feat2="primary_practitioner_id", fname="dh", L=0, validate=F , calc_Mean_Max=T)
# "primary_physician_role primary_practitioner_id 0.752364317274297"
# "primary_physician_role primary_practitioner_id 0.769226314565444"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="primary_practitioner_id",feat2="diagnosis_date", fname="dh", L=0, validate=F , calc_Mean_Max=T)
# "primary_practitioner_id diagnosis_date 0.7215431758733"
# "primary_practitioner_id diagnosis_date 0.690001524211778"
LIKELIHOOD_2WAY_CV_MEAN_MAX( feat1="primary_practitioner_id",feat2="pid_dh1", fname="dh", L=0, validate=F , calc_Mean_Max=T)
# "primary_practitioner_id pid_dh1 0.74767898101251"
# "primary_practitioner_id pid_dh1 0.762341578202555"

build3 <- copy(bas)
fn.save.data("build3")
rm(build3);gc()
########################################################################
