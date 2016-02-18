rm(list=ls())
source("fn.base.r");gc()

fn.load.data("build1")
bas <- copy(build1)
rm(build1);gc()

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

dt1[, claim_type:=NULL]

dt2 <- fread('../data/input/physicians.csv', header=T );gc()
setkeyv( dt2 , "practitioner_id"  )
head(dt2)
dt1[, pid_1 := dt2[J(dt1$practitioner_id)]$physician_id ]
dt1[, pid_2 := dt2[J(dt1$practitioner_id)]$state ]
dt1[, pid_3 := dt2[J(dt1$practitioner_id)]$specialty_code ]
dt1[, pid_4 := dt2[J(dt1$practitioner_id)]$specialty_description ]
dt1[, pid_5 := dt2[J(dt1$practitioner_id)]$CBSA ]
rm(dt2);gc()


dt1[, cv := bas[J(dt1$patient_id)]$cv ];gc()
dt1[, tgt:= bas[J(dt1$patient_id)]$is_screener ];gc()

LIKELIHOOD_CV_MEAN( feat="surgical_code", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="place_of_service", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="plan_type", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="practitioner_id", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="primary_physician_role", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="sc1", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="sc2", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="sc3", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="pid_1", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="pid_2", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="pid_3", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="pid_4", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="pid_5", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="claim_id", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="procedure_type_code", fname="sh",  validate=F )
LIKELIHOOD_CV_MEAN( feat="surgical_procedure_date", fname="sh",  validate=F )

LIKELIHOOD_CV_MAX( feat="surgical_code", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="place_of_service", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="plan_type", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="practitioner_id", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="primary_physician_role", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="sc1", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="sc2", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="sc3", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="pid_1", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="pid_2", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="pid_3", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="pid_4", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="pid_5", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="claim_id", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="procedure_type_code", fname="sh",  validate=F )
LIKELIHOOD_CV_MAX( feat="surgical_procedure_date", fname="sh",  validate=F )

rm(dt1,dt2,dt);gc()
########################################################################


########################################################################
dt1 <- fread('../data/input/diagnosis_head.csv', header=T );gc()
setkeyv( dt1, c("patient_id","diagnosis_date")  );gc()

ind <- which( dt1$patient_id %in% bas$patient_id );gc()
dt1 <- dt1[ ind ];gc()

dt <- dt1[, .N  , keyby="patient_id"];gc()
bas[, f3_N := dt[J(bas$patient_id)]$N ];gc()

dt <- dt1[, sum( primary_physician_role=="ATG",na.rm=T )  , keyby="patient_id"];gc()
bas[, f3_0 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum( primary_physician_role=="RND",na.rm=T )  , keyby="patient_id"];gc()
bas[, f3_1 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum( primary_physician_role=="",na.rm=T )  , keyby="patient_id"];gc()
bas[, f3_2 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum( primary_physician_role=="PRV",na.rm=T )  , keyby="patient_id"];gc()
bas[, f3_3 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum( primary_physician_role=="ORD",na.rm=T )  , keyby="patient_id"];gc()
bas[, f3_4 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum( primary_physician_role=="UNK",na.rm=T )  , keyby="patient_id"];gc()
bas[, f3_5 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum( primary_physician_role=="OTH",na.rm=T )  , keyby="patient_id"];gc()
bas[, f3_6 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum( primary_physician_role=="OPR",na.rm=T )  , keyby="patient_id"];gc()
bas[, f3_7 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum( claim_type=="HX",na.rm=T )  , keyby="patient_id"];gc()
bas[, f3_8 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum( claim_type=="MX",na.rm=T )  , keyby="patient_id"];gc()
bas[, f3_9 := dt[J(bas$patient_id)]$V1 ];gc()

dt1[, claim_type := NULL ] ; gc()
dt1[, primary_physician_role := NULL ] ; gc()

dt <- dt1[, length(unique(claim_id))  , keyby="patient_id"];gc()
bas[, f3_10 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(diagnosis_code))  , keyby="patient_id"];gc()
bas[, f3_11 := dt[J(bas$patient_id)]$V1 ];gc()

bas1 <- copy(bas[, colnames(bas)[c(1,7,9)], with=F ]);gc()
setkeyv( bas1, "patient_id"  )

dt1[, cv := bas1[J(dt1$patient_id)]$cv ];gc()
dt1[, tgt:= bas1[J(dt1$patient_id)]$is_screener ];gc()

LIKELIHOOD_CV_MAX( "diagnosis_code" , fname="dh", validate=F );gc()
LIKELIHOOD_CV_MAX( "primary_practitioner_id" , fname="dh", validate=F );gc()

LIKELIHOOD_CV_MEAN( "diagnosis_code" , fname="dh", validate=F );gc()
LIKELIHOOD_CV_MEAN( "primary_practitioner_id" , fname="dh", validate=F );gc()


dt <- dt1[ , .N, by="diagnosis_code"]

tmp <- dt$diagnosis_code;gc()
tmp1 <- substr( tmp , 1,1 )

ind1 <- which( tmp1 == "E"  );gc()
tmp[ ind1 ] <- substr( tmp[ ind1 ], 2,7 );gc()

ind2 <- which( tmp1 == "V"  );gc()
tmp[ ind2 ] <- substr( tmp[ ind2 ], 2,7 );gc()

ind3 <- which( tmp1 == "N"  );gc()
tmp[ ind3 ] <- substr( tmp[ ind3 ], 2,7 );gc()


indf <- which( grepl("\\.", tmp ) == F );gc()
length(indf)
tmp[ indf  ] <- paste0(tmp[indf], ".00" )
tmps1 <- unlist( strsplit( tmp , "\\."  ) )[ seq(1,2*length(tmp),2 ) ]
tmps2 <- unlist( strsplit( tmp , "\\."  ) )[ seq(2,2*length(tmp),2 ) ]
dt$s1 <- tmps1
dt$s2 <- tmps2
dt$ind <- 0
dt$ind[ind1] <- 1
dt$ind[ind2] <- 2
dt$ind[ind3] <- 4
setkeyv( dt, "diagnosis_code"  )


dt1[, diagnosis_code_ind := dt[J(dt1$diagnosis_code)]$ind  ];gc()
dt1[, diagnosis_code_sub := dt[J(dt1$diagnosis_code)]$s2  ];gc()
dt1[, diagnosis_code := dt[J(dt1$diagnosis_code)]$s1  ];gc()


LIKELIHOOD_CV_MAX( "diagnosis_code" , fname="dh2", validate=F );gc()
LIKELIHOOD_CV_MAX( "diagnosis_code_sub" , fname="dh", validate=F );gc()
LIKELIHOOD_CV_MAX( "diagnosis_code_ind" , fname="dh", validate=F );gc()
LIKELIHOOD_CV_MEAN( "diagnosis_code" , fname="dh2", validate=F );gc()
LIKELIHOOD_CV_MEAN( "diagnosis_code_sub" , fname="dh", validate=F );gc()
LIKELIHOOD_CV_MEAN( "diagnosis_code_ind" , fname="dh", validate=F );gc()


dt1[, claim_id:=NULL ];gc()
dt1[, diagnosis_date:=NULL ];gc()
dt1[, diagnosis_code:=NULL ];gc()

dt2 <- fread('../data/input/physicians.csv', header=T );gc()
setkeyv( dt2 , "practitioner_id"  )

dt1[, pid_dh1 := dt2[J(dt1$primary_practitioner_id)]$physician_id ];gc()
dt1[, pid_dh2 := dt2[J(dt1$primary_practitioner_id)]$state ];gc()
dt1[, pid_dh3 := dt2[J(dt1$primary_practitioner_id)]$specialty_code ];gc()
dt1[, pid_dh4 := dt2[J(dt1$primary_practitioner_id)]$CBSA ];gc()
rm(dt2);gc()


LIKELIHOOD_CV_MAX( "pid_dh1" , fname="ph", validate=F );gc()
LIKELIHOOD_CV_MEAN( "pid_dh1" , fname="ph", validate=F );gc()
# [1] "pid_dh1 0.749893823125735"
# [1] "pid_dh1 0.728748414046949"

LIKELIHOOD_CV_MAX( "pid_dh2" , fname="ph", validate=F );gc()
LIKELIHOOD_CV_MEAN( "pid_dh2" , fname="ph", validate=F );gc()
# [1] "pid_dh2 0.561090649276746"
# [1] "pid_dh2 0.599824307637348"

LIKELIHOOD_CV_MAX( "pid_dh3" , fname="ph", validate=F );gc()
LIKELIHOOD_CV_MEAN( "pid_dh3" , fname="ph", validate=F );gc()
# [1] "pid_dh3 0.685131846450797"
# [1] "pid_dh3 0.686268044012761"

LIKELIHOOD_CV_MAX( "pid_dh4" , fname="ph", validate=F );gc()
LIKELIHOOD_CV_MEAN( "pid_dh4" , fname="ph", validate=F );gc()
# [1] "pid_dh4 0.588853828684764"
# [1] "pid_dh4 0.644936654377402"

rm(dt1,dt,ind);gc()
########################################################################



########################################################################
dt1 <- fread('../data/input/prescription_head.csv', header=T );gc()
setorder( dt1, patient_id, rx_fill_date );gc()

bas1 <- copy(bas[, colnames(bas)[c(1,7,9)], with=F ]);gc()
setkeyv( bas1, "patient_id"  )
dt1[, cv := bas1[J(dt1$patient_id)]$cv ];gc()
dt1[, tgt:= bas1[J(dt1$patient_id)]$is_screener ];gc()

ind <- which( dt1$patient_id %in% bas$patient_id );gc()
dt1 <- dt1[ ind ];gc()

dt <- dt1[, .N  , keyby="patient_id"];gc()
bas[, f4_N := dt[J(bas$patient_id)]$N ]
dt <- dt1[, length(unique( payment_type   ))  , keyby="patient_id"];gc()
bas[, f4_1 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum( payment_type=="COMMERCIAL",na.rm=T )  , keyby="patient_id"];gc()
bas[, f4_2 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum( payment_type=="CASH",na.rm=T )  , keyby="patient_id"];gc()
bas[, f4_3 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum( payment_type=="MANAGED MEDICAID",na.rm=T )  , keyby="patient_id"];gc()
bas[, f4_4 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum( payment_type=="MEDICAID",na.rm=T )  , keyby="patient_id"];gc()
bas[, f4_5 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum( payment_type=="MEDICARE",na.rm=T )  , keyby="patient_id"];gc()
bas[, f4_6 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, sum( payment_type=="ASSISTANCE PROGRAMS",na.rm=T )  , keyby="patient_id"];gc()
bas[, f4_7 := dt[J(bas$patient_id)]$V1 ]

dt <- dt1[, length(unique(claim_id))  , keyby="patient_id"];gc()
bas[, f4_8 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, length(unique(drug_id))  , keyby="patient_id"];gc()
bas[, f4_9 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, length(unique(refill_code))  , keyby="patient_id"];gc()
bas[, f4_10 := dt[J(bas$patient_id)]$V1 ]
dt <- dt1[, mean(days_supply,na.rm=T)  , keyby="patient_id"];gc()
bas[, f4_11 := dt[J(bas$patient_id)]$V1 ]


LIKELIHOOD_CV_MAX( feat="drug_id" ,fname="pch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="drug_id" ,fname="pch", validate=F );gc()
# [1] "drug_id 0.604918114415133"
# [1] "drug_id 0.637804398733077"

LIKELIHOOD_CV_MAX( feat="practitioner_id" ,fname="pch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="practitioner_id" ,fname="pch", validate=F );gc()
# [1] "practitioner_id 0.645067569112733"
# [1] "practitioner_id 0.687703664648013"

LIKELIHOOD_CV_MAX( feat="refill_code"   ,fname="pch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="refill_code"   ,fname="pch", validate=F );gc()
# [1] "refill_code 0.499383802253701"
# [1] "refill_code 0.509408442359961"

LIKELIHOOD_CV_MAX( feat="days_supply"   ,fname="pch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="days_supply"   ,fname="pch", validate=F );gc()
# [1] "days_supply 0.527947172366406"
# [1] "days_supply 0.569801745355208"

dt1[, claim_id := NULL]
dt1[, practitioner_id := NULL]
dt1[, refill_code := NULL]
dt1[, days_supply := NULL]
dt1[, rx_fill_date := NULL]
dt1[, rx_number := NULL]
dt1[, payment_type := NULL]
gc()

dt2 <- fread('../data/input/drugs.csv', header=T );gc()
setkeyv( dt2 , "drug_id"  )

dt1[, drug_1 := dt2[J(dt1$drug_id)]$drug_name ];gc()
dt1[, drug_2 := dt2[J(dt1$drug_id)]$BGI ];gc()
dt1[, drug_3 := dt2[J(dt1$drug_id)]$BB_USC_name ];gc()
dt1[, drug_4 := dt2[J(dt1$drug_id)]$drug_strength ];gc()

LIKELIHOOD_CV_MAX( feat="drug_1"   ,fname="dg", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="drug_1"   ,fname="dg", validate=F );gc()
# [1] "drug_1 0.604494910814994"
# [1] "drug_1 0.627703507020975"

LIKELIHOOD_CV_MAX( feat="drug_2"   ,fname="dg", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="drug_2"   ,fname="dg", validate=F );gc()
# [1] "drug_2 0.499771168771082"
# [1] "drug_2 0.50864210789468"

LIKELIHOOD_CV_MAX( feat="drug_3"   ,fname="dg", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="drug_3"   ,fname="dg", validate=F );gc()
# [1] "drug_3 0.593204998587821"
# [1] "drug_3 0.617546395232292"

LIKELIHOOD_CV_MAX( feat="drug_4"   ,fname="dg", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="drug_4"   ,fname="dg", validate=F );gc()
# [1] "drug_4 0.591655349338194"
# [1] "drug_4 0.620557724348882"

rm(dt1,dt,ind);gc()
########################################################################


########################################################################
dt1 <- fread('../data/input/procedure_head.csv', header=T );gc()
setorder( dt1, patient_id );gc()

ind <- which( dt1$patient_id %in% bas$patient_id );gc()
dt1 <- dt1[ ind ];gc()

bas1 <- copy(bas[, colnames(bas)[c(1,7,9)], with=F ]);gc()
setkeyv( bas1, "patient_id"  )
dt1[, cv := bas1[J(dt1$patient_id)]$cv ];gc()
dt1[, tgt:= bas1[J(dt1$patient_id)]$is_screener ];gc()

dt <- dt1[, .N , keyby="patient_id"];gc()
bas[, f5_1 := dt[J(bas$patient_id)]$N ];gc()
dt <- dt1[, length(unique(claim_id)) , keyby="patient_id"];gc()
bas[, f5_2 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(claim_line_item)) , keyby="patient_id"];gc()
bas[, f5_3 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(claim_type)) , keyby="patient_id"];gc()
bas[, f5_4 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(procedure_code)) , keyby="patient_id"];gc()
bas[, f5_5 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(place_of_service)) , keyby="patient_id"];gc()
bas[, f5_6 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(plan_type)) , keyby="patient_id"];gc()
bas[, f5_7 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(primary_practitioner_id)) , keyby="patient_id"];gc()
bas[, f5_8 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum(units_administered,na.rm=T) , keyby="patient_id"];gc()
bas[, f5_9 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum(charge_amount,na.rm=T) , keyby="patient_id"];gc()
bas[, f5_10 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(primary_physician_role)) , keyby="patient_id"];gc()
bas[, f5_11 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(attending_practitioner_id)) , keyby="patient_id"];gc()
bas[, f5_12 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(referring_practitioner_id)) , keyby="patient_id"];gc()
bas[, f5_13 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(rendering_practitioner_id)) , keyby="patient_id"];gc()
bas[, f5_14 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(ordering_practitioner_id)) , keyby="patient_id"];gc()
bas[, f5_15 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, length(unique(operating_practitioner_id)) , keyby="patient_id"];gc()
bas[, f5_16 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum(claim_type=="HX",na.rm=T) , keyby="patient_id"];gc()
bas[, f5_17 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum(claim_type=="MX",na.rm=T) , keyby="patient_id"];gc()
bas[, f5_18 := dt[J(bas$patient_id)]$V1 ];gc()

dt <- dt1[, sum(primary_physician_role=="ATG",na.rm=T) , keyby="patient_id"];gc()
bas[, f5_19 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum(primary_physician_role=="RND",na.rm=T) , keyby="patient_id"];gc()
bas[, f5_20 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum(primary_physician_role=="",na.rm=T) , keyby="patient_id"];gc()
bas[, f5_21 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum(primary_physician_role=="PRV",na.rm=T) , keyby="patient_id"];gc()
bas[, f5_22 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum(primary_physician_role=="ORD",na.rm=T) , keyby="patient_id"];gc()
bas[, f5_23 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum(primary_physician_role=="UNK",na.rm=T) , keyby="patient_id"];gc()
bas[, f5_24 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum(primary_physician_role=="OTH",na.rm=T) , keyby="patient_id"];gc()
bas[, f5_25 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, sum(primary_physician_role=="OPR",na.rm=T) , keyby="patient_id"];gc()
bas[, f5_26 := dt[J(bas$patient_id)]$V1 ];gc()

dt <- dt1[, min(procedure_date) , keyby="patient_id"];gc()
bas[, f5_27 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, max(procedure_date) , keyby="patient_id"];gc()
bas[, f5_28 := dt[J(bas$patient_id)]$V1 ];gc()
bas[, f5_29 := (f5_28-f5_27)/f5_1 ];gc()

dt <- dt1[, max(charge_amount,na.rm=T) , keyby="patient_id"];gc()
bas[, f5_30 := dt[J(bas$patient_id)]$V1 ];gc()
dt <- dt1[, max(units_administered,na.rm=T) , keyby="patient_id"];gc()
bas[, f5_31 := dt[J(bas$patient_id)]$V1 ];gc()


LIKELIHOOD_CV_MAX( feat="claim_line_item"   ,fname="proch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="claim_line_item"   ,fname="proch", validate=F );gc()
# [1] "claim_line_item 0.500036644976482"
# [1] "claim_line_item 0.535972894582423"

LIKELIHOOD_CV_MAX( feat="procedure_code"  ,fname="proch" , validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="procedure_code"  ,fname="proch" , validate=F );gc()
# [1] "procedure_code 0.737623124309815"
# [1] "procedure_code 0.731781019621885"

LIKELIHOOD_CV_MAX( feat="place_of_service"  ,fname="proch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="place_of_service"  ,fname="proch", validate=F );gc()
# [1] "place_of_service 0.506761760489579"
# [1] "place_of_service 0.535200286925995"

LIKELIHOOD_CV_MAX( feat="plan_type"  ,fname="proch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="plan_type"  ,fname="proch", validate=F );gc()
# [1] "plan_type 0.511710934379664"
# [1] "plan_type 0.568342530832602"

LIKELIHOOD_CV_MAX( feat="primary_practitioner_id" ,fname="proch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="primary_practitioner_id" ,fname="proch", validate=F );gc()
# [1] "primary_practitioner_id 0.7277436184106"
# [1] "primary_practitioner_id 0.743241044238176"

# LIKELIHOOD_CV_MAX( feat="attending_practitioner_id" ,fname="proch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="attending_practitioner_id" ,fname="proch", validate=F );gc()
# [1] "attending_practitioner_id 0.696922170292819"
# [1] "attending_practitioner_id 0.710253870248606"

# LIKELIHOOD_CV_MAX( feat="referring_practitioner_id"   ,fname="proch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="referring_practitioner_id"   ,fname="proch", validate=F );gc()
# [1] "referring_practitioner_id 0.695701482893676"
# [1] "referring_practitioner_id 0.693703151033367"

LIKELIHOOD_CV_MAX( feat="rendering_practitioner_id"   ,fname="proch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="rendering_practitioner_id"   ,fname="proch", validate=F );gc()
# [1] "rendering_practitioner_id 0.729572624902439"
# [1] "rendering_practitioner_id 0.72737098745953"

LIKELIHOOD_CV_MAX( feat="ordering_practitioner_id"   ,fname="proch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="ordering_practitioner_id"   ,fname="proch", validate=F );gc()
# [1] "ordering_practitioner_id 0.503967176389742"
# [1] "ordering_practitioner_id 0.522081998347678"

LIKELIHOOD_CV_MAX( feat="operating_practitioner_id"   ,fname="proch", validate=F );gc()
LIKELIHOOD_CV_MEAN( feat="operating_practitioner_id"   ,fname="proch", validate=F );gc()
# [1] "operating_practitioner_id 0.591953610563421"
# [1] "operating_practitioner_id 0.606914270935211"

gc()
dim(bas)
head(bas)
colnames(bas)

build2 <- copy(bas)
fn.save.data("build2")
rm(build2);gc()
########################################################################