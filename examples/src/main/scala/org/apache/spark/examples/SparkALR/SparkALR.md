# SparkALR


![ALR](fastals.png)

## Compilation

To compile and run, run the following from the Spark root directory. Compilation:
```
sbt/sbt assembly
```
To run with 4GB of ram:
```
./bin/spark-submit --class org.apache.spark.mllib.examples.SparkALR  \
  ./examples/target/scala-2.10/spark-examples-1.1.0-SNAPSHOT-hadoop1.0.4.jar \
  --executor-memory 4G \
  --driver-memory 4G
```

# ALR

<!-- For example, the following code fits a model outputting `ms` and `us` as the factors:

    // Iteratively update movies then users
    for (iter <- 1 to ITERATIONS) {
      println("Iteration " + iter + ":")

      // Update ms
      println("Computing new ms")
      ms = multByXstar(R, ms, us, minimizer(us))

      // Update us
      println("Computing new us")
      us = multByXstarTranspose(Rt, ms, us, minimizer(ms))
    } -->
