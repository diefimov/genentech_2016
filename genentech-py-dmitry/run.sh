# create s3 bucket with the name genentech-2016

# install package awscli for Python
#sudo pip install --upgrade awscli
#complete -C aws_completer aws

#aws configure
# AWS Access Key ID [None]: [from Security Credentials]
# AWS Secret Access Key [None]: [from Security Credentials]
# Default region name [None]: us-west-2
# Default output format [None]: json

# check the files on s3 bucket
#aws s3 ls s3://genentech-2016

# copy files to s3 bucket
#aws s3 cp diagnosis_code.csv.gz s3://genentech-2016
#aws s3 cp diagnosis_head.csv.gz s3://genentech-2016
#aws s3 cp drugs_head.csv.gz s3://genentech-2016
#aws s3 cp patient_activity_head.csv.gz s3://genentech-2016
#aws s3 cp patients_test.csv.gz s3://genentech-2016
#aws s3 cp patients_train.csv.gz s3://genentech-2016
#aws s3 cp physicians.csv.gz s3://genentech-2016
#aws s3 cp prescription_head.csv.gz s3://genentech-2016
#aws s3 cp procedure_code.csv.gz s3://genentech-2016
#aws s3 cp procedure_head.csv.gz s3://genentech-2016
#aws s3 cp surgical_code.csv.gz s3://genentech-2016
#aws s3 cp surgical_head.csv.gz s3://genentech-2016

# run the python code to copy files from s3 bucket to Amazon Redshift
python copy_from_s3_to_redshift.py
python generate_train_test_without_excluded_patients.py
python generate_cv_table.py --nfold 3
python clean_diagnosis_code_table.py
python generate_feature_tables.py


# check disk usage on amazon redshift (in SQL Workbench/J)
#select owner, host, diskno, used, capacity,
#(used-tossed)/capacity::numeric *100 as pctused 
#from stv_partitions order by owner;

# build features
python build_features.py