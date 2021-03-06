/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.examples

import collection.mutable.HashMap
import org.apache.commons.math3.linear._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.HashPartitioner
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.{ StructType, StructField, LongType, DoubleType };
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.linalg.{ Vector, Vectors, VectorUDT }
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.{ SparkConf, SparkContext }
import scala.util._

/**
 * SparkALR for Spark
 */
object SparkALR {
  // Number of users
  val U = 5000
  // Number of movies
  val M = 100
  // Number of features
  val F = 5
  // Number of iterations
  val ITERATIONS = 5
  // Number of regression iterations
  val REGMAXITER = 2
  // Regularization parameter
  val REGP = 1e-4
  // Elastic-net parameter
  val ENET = 0.00
  // Number of partitions for data (set to number of machines in cluster)
  val NUMPARTITIONS = 4
  // File name to read data
  val fileName = "data/mllib/SparkALR.data.nu5000_nm100_nl10_k2_N10000.csv"
  val outputDir = "./"

  // scala context that is visible to all in SparkALR
  val sparkConf = new SparkConf().setAppName("SparkALR")
  val sc = new SparkContext(sparkConf)
  val sqlContext = new SQLContext(sc)

  private def formSuffStat(data: RDD[((Long, Long), (Double, Int))], link: String) :
    RDD[(Long, HashMap[Long,(Double, Double)])] = link match {
    // RDD[(u, HashMap[m,(t, n)])]
    case "logistic" => val initialMap = HashMap.empty[Long, (Double, Double)]
                       val addToMap = (s: HashMap[Long, (Double, Double)],
                                       v: (Long, (Double, Double))) => s+= (v._1 -> v._2)
                       val mergePartitionMap = (p1: HashMap[Long, (Double, Double)],
                                                p2: HashMap[Long, (Double, Double)]) => p1 ++= p2
    data.reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2)).mapValues(v => (v._1, v._2-v._1)).
      map(v => (v._1._1, (v._1._2, v._2))).aggregateByKey(initialMap)(addToMap, mergePartitionMap)
  }

  private def makeTrainDF_u(m_id: Long,
                            um_us: RDD[(Long, (HashMap[Long,(Double, Double)], Vector))]) :
                            DataFrame = {
    // Generate the schema based on the string of schema
    val schema =
        StructType(
          StructField("label", DoubleType, true) ::
          StructField("weight", DoubleType, true) ::
          StructField("features", new VectorUDT, true) :: Nil)
    val um_us_im = um_us.filter(v => v._2._1.contains(m_id)).
      mapValues(v => (v._1(m_id), v._2)).
      flatMap(v => Seq((1.0, v._2._1._1, v._2._2), (0.0, v._2._1._2, v._2._2))).
      map(v => Row(v._1, v._2, v._3))
    sqlContext.createDataFrame(um_us_im, schema)
  }

  private def update_us(lr: LogisticRegression, data_u: HashMap[Long,(Double, Double)], ms: Array[Vector]) : Vector = {
    val mu_features = data_u.keySet.toArray.map(v => v.toInt -1) collect ms
    val u_instance = data_u.values.toArray.zipWithIndex.flatMap(v => Seq((1.0, v._1._1, mu_features(v._2)), (0.0, v._1._2, mu_features(v._2))))
    lr.localTrain(u_instance).coefficients.toDense
  }

  def main(args: Array[String]) {
    sc.setLogLevel("WARN")

    printf("Running with M=%d, U=%d, rank=%d, iters=(%d, %d), reg=(%f, %f)\n",
      M, U, F, ITERATIONS, REGMAXITER, REGP, ENET)

    printf("Reading file from %s \n.", fileName)
    val t0 = System.nanoTime()
    // Create data in the form of RDD[((Long, Long), (Double, Int))]
    val data = sc.textFile(fileName).map(_.split(",")).
                map(v => ((v(0).toLong, v(1).toLong), (v(2).toDouble, 1)))

    // *** row indexed sufficicnet stat data
    //  RDD[(Long, HashMap[Long,(Double, Double)])]
    val um_data = formSuffStat(data, "logistic").partitionBy(new HashPartitioner(NUMPARTITIONS))

    // *** row index user features
    //  RDD[(Long, Vector)]
    var us = um_data.mapValues(v => Vectors.dense(Array.fill(F)(math.random-0.5)))

    // *** column index movie features
    //  Array[Vector]
    var ms = Array.fill(M)(Vectors.dense(Array.fill(F)(math.random-0.5)))
    var msb = sc.broadcast(ms)

    var um_us = um_data.join(us)
    // *** LogisticRegression models for both distributed and local calculation
    val lr_u = new LogisticRegression()
                    .setMaxIter(REGMAXITER)
                    .setRegParam(REGP)
                    .setElasticNetParam(ENET)
                    .setFitIntercept(false)
                    .setStandardization(false)

    val lr_m = new LogisticRegression()
                    .setMaxIter(REGMAXITER)
                    .setRegParam(REGP)
                    .setElasticNetParam(ENET)
                    .setFitIntercept(false)
                    .setStandardization(false)
                    .setWeightCol("weight")

    val t1 = System.nanoTime()
    for (iter <- 1 to ITERATIONS) {
      println("Iteration " + iter + ":")

      // *** Update ms *** //
      println("Update ms")

      //  join data with us
      um_us = um_data.join(us)
      //  loop over entries of ms
      for( m_id <- 1 to M ){
        ms(m_id-1) = lr_m.fit(makeTrainDF_u(m_id, um_us)).coefficients.toDense
      }

      // *** Update us *** //
      println("Update us")

      //  broadcast ms
      msb = sc.broadcast(ms)

      // map the local trainer with data
      us = um_data.mapValues(v => update_us(lr_u, v, msb.value))

    }
    val t2 = System.nanoTime()
    println("t1 - t0: " + (t1 - t0)/1.0e9 + "sec", ", t2 - t1:" + (t2 - t1)/ITERATIONS/1.0e9 + "sec")
    // write ouput
    us.coalesce(1,true).saveAsTextFile(outputDir + "us.csv")
    sc.parallelize(ms).coalesce(1,true).saveAsTextFile(outputDir + "ms.csv")
    sc.stop()
  }
}
