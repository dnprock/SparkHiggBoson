SparkHiggBoson
==============

Spark ML for Kaggle Higg Boson challenge

Download data file from Kaggle site:

http://www.kaggle.com/c/higgs-boson/data

Remove weight column in training.csv and rename it to training-noweight.csv.

Compile:

sbt package

Run:

SPARK_HOME/bin/spark-submit  --class "KaggleHiggBosonLabel" --master spark://host:7077 target/scala-2.10/kaggle-higg-boson_2.10-1.0.jar
