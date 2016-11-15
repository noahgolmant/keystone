package workflow

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.FunSuite
import pipelines.Logging

class EstimatorSuite extends FunSuite with PipelineContext with Logging {
  test("Estimator fit RDD") {
    sc = new SparkContext("local", "test")

    val intEstimator = new Estimator[Int, Int] {
      def fit(data: RDD[Int]): Transformer[Int, Int] = {
        val first = data.first()
        Transformer(x => x + first)
      }
    }

    val trainData = sc.parallelize(Seq(32, 94, 12))
    val testData = sc.parallelize(Seq(42, 58, 61))

    val pipeline = intEstimator.withData(trainData)
    assert(pipeline.apply(testData).get().collect().toSeq === Seq(42 + 32, 58 + 32, 61 + 32))
  }

  test("Estimator fit Pipeline Data") {
    sc = new SparkContext("local", "test")

    val transformer = Transformer[Int, Int](_ * 2)

    val intEstimator = new Estimator[Int, Int] {
      def fit(data: RDD[Int]): Transformer[Int, Int] = {
        val first = data.first()
        Transformer(x => x + first)
      }
    }

    val trainData = sc.parallelize(Seq(32, 94, 12))
    val testData = sc.parallelize(Seq(42, 58, 61))

    val pipeline = intEstimator.withData(transformer(trainData))
    assert(pipeline.apply(testData).get().collect().toSeq === Seq(42 + 64, 58 + 64, 61 + 64))
  }
  
  test("Test pipeline withData fitting") {
    sc = new SparkContext("local", "test")

    val transformer = Transformer[Int, Int](_ * 2)

    val intEstimator = new Estimator[Int, Int] {
      def fit(data: RDD[Int]): Transformer[Int, Int] = {
        val last = data.collect().last
        Transformer(x => x + last)
      }
    }

    val trainData = sc.parallelize(Seq(12, 94, 32))
    val trainDataTwo = sc.parallelize(Seq(15, 10, 5))
    val testData = sc.parallelize(Seq(42, 58, 61))

    val pipeline = intEstimator.withData(transformer(trainData))
    val newPipeline = pipeline.withData(intEstimator, PipelineDataset(transformer(trainDataTwo))).toPipeline
    val results = newPipeline.apply(testData)
    assert(results.get().collect().toSeq === Seq(42 + 10, 58 + 10, 61 + 10))
  }

  test("Test incremental estimator withData") {
    sc = new SparkContext("local", "test")

    val transformer = Transformer[Int, Int](_ * 2)

    val intEstimator = new IncrementalEstimator[Int, Int, Int] {
      override def fit(data: RDD[Int], oldModel: Int): Int = data.collect().last + oldModel

      override def transformer(model: Int): Transformer[Int, Int] = Transformer(x => x + model)
    }

    val trainData = sc.parallelize(Seq(12, 94, 32))
    val trainDataTwo = sc.parallelize(Seq(15, 10, 5))
    val testData = sc.parallelize(Seq(42, 58, 61))

    val (pipeline, firstModel) = intEstimator.withData(PipelineDataset(transformer(trainData)), PipelineDatum(0))
    assert(firstModel.get == 64)

    val (newPipeline, secondModel) = intEstimator.withData(PipelineDataset(transformer(trainData)), firstModel)
    assert(secondModel.get == 64*2)
    assert(pipeline.apply(testData).get().collect().toSeq === Seq(42 + 64, 58 + 64, 61 + 64))
    assert(newPipeline.apply(testData).get().collect().toSeq === Seq(42 + 64*2, 58 + 64*2, 61 + 64*2))


  }

}
