## About this document

This document describes how to reproduce the prediction files and feature files.

## Requirements

I used following software and packages:

* Anaconda 2.4.1 (bundle of python packages for data analytics)
  * Python 2.7.11
  * bloscpack 0.10.0
  * blosc 1.2.5
  * numpy 1.10.1
  * scipy 0.16.0
  * sklearn 0.16.0
* xgboost 0.4

For reference, Dockerfile for setting up the environment is also available
on the `genentech-py-kohei/base/` directory.

## File structures

I assumed that all files on `genentech-py-kohei/` are deployed on the working directory as below:

```bash
$ tree
.
├── genentech_path.py
├── gen_feat.py
├── gen_feat2.py
└── kohei_v52.py
```

All prediction files are stored on `data/team/kohei/`.
Generated features are stored on `genentech-py-kohei/`.

## Instruction for reproducing prediction files and feature files

Note that Kohei's model requires huge amount of computational resources.
It requires 160 GB RAM and expected runtime is about 30 hours with 36 cores of
CPUs. Amazon EC2 m4.10xlarge instance is suitable for computing this.

```bash
# Generate features and store them on `data/working`.
$ python gen_feat.py

# Generate prediction file of v52 model for each fold and merge those results
# - Output: data/team/kohei/kohei_v52_prediction_train.csv.gz
# - Output: data/team/kohei/kohei_v52_prediction_test_full.csv.gz
$ python kohei_v52.py --validate --fold1
$ python kohei_v52.py --validate --fold2
$ python kohei_v52.py --validate --fold3
$ python kohei_v52.py --test
$ python kohei_v52.py --merge

# Generate features of missing claims.
# - Output: data/team/kohei/train_num_removed_claim_id_*.csv.gz
# - Output: data/team/kohei/test_num_removed_claim_id_*.csv.gz
$ python gen_feat2.py
```
