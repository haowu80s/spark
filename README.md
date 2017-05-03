# SparkALR

Alternating logistic regression is a collaborative filtering method for the prediction of occurance probability given binary observations (e.g. click-though rate).

## Compilation
To compile:

    sbt/sbt assembly

## Run
To run with 4GB of ram:

    ./bin/spark-submit --class org.apache.spark.ml.examples.SparkALR \
        ./examples/target/scala-2.10/spark-examples-1.6.2-SNAPSHOT-hadoop2.2.0.jar \
        --executor-memory 4G  --driver-memory 4G
    
## Implementations
Detailed information can be found [here](./examples/src/main/scala/org/apache/spark/examples/SparkALR/).

All the implementations are in the SparkALR.scala except the localTrain method for LogisticRegression() which is in LogisiticRegression.scala.

Sample data is included at data/mllib/SparkALR.data.csv
