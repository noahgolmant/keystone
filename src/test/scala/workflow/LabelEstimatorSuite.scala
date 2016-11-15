package workflow

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.FunSuite
import pipelines.Logging

class LabelEstimatorSuite extends FunSuite with PipelineContext with Logging {
  test("LabelEstimator fit RDD") {
    sc = new SparkContext("local", "test")

    val intEstimator = new LabelEstimator[Int, Int, String] {
      def fit(data: RDD[Int], labels: RDD[String]): Transformer[Int, Int] = {
        val first = data.first()
        val label = labels.first().hashCode
        Transformer(x => x + first + label)

      }
    }

    val trainData = sc.parallelize(Seq(32, 94, 12))
    val trainLabels = sc.parallelize(Seq("sjkfdl", "iw", "432"))
    val testData = sc.parallelize(Seq(42, 58, 61))

    val pipeline = intEstimator.withData(trainData, trainLabels)
    val offset = 32 + "sjkfdl".hashCode
    assert(pipeline.apply(testData).get().collect().toSeq === Seq(42 + offset, 58 + offset, 61 + offset))
  }

  test("LabelEstimator fit pipeline data") {
    sc = new SparkContext("local", "test")

    val dataTransformer = Transformer[Int, Int](_ * 2)
    val labelTransformer = Transformer[String, String](_ + "hi")

    val intEstimator = new LabelEstimator[Int, Int, String] {
      def fit(data: RDD[Int], labels: RDD[String]): Transformer[Int, Int] = {
        val first = data.first()
        val label = labels.first().hashCode
        Transformer(x => x + first + label)

      }
    }

    val trainData = sc.parallelize(Seq(32, 94, 12))
    val trainLabels = sc.parallelize(Seq("sjkfdl", "iw", "432"))
    val testData = sc.parallelize(Seq(42, 58, 61))

    val pipeline = intEstimator.withData(dataTransformer(trainData), labelTransformer(trainLabels))
    val offset = 64 + "sjkfdlhi".hashCode
    assert(pipeline.apply(testData).get().collect().toSeq === Seq(42 + offset, 58 + offset, 61 + offset))
  }

  test("IncrementalLabelEstimator fit pipeline data") {
    sc = new SparkContext("local", "test")

    val dataTransformer = Transformer[Int, Int](_ * 2)
    val labelTransformer = Transformer[String, String](_ + "hi")


    var hash = 0

    val intEstimator = new IncrementalLabelEstimator[Int, Int, String, Int] {
      override def fit(data: RDD[Int], labels: RDD[String], oldModel: Int): Int = {
        val first = data.first()
        val label = labels.first().hashCode()
        hash = label
        first + label + oldModel
      }

      override def transformer(model: Int): Transformer[Int, Int] = Transformer(x => x + model)
    }

    val trainData = sc.parallelize(Seq(32, 94, 12))
    val trainLabels = sc.parallelize(Seq("sjkfdl", "iw", "432"))
    val testData = sc.parallelize(Seq(42, 58, 61))

    val (pipeline, firstModel) = intEstimator.withData(dataTransformer(trainData), labelTransformer(trainLabels), 0)

    val firstOffset = 64 + hash
    assert(firstModel.get == firstOffset)
    val firstResults = pipeline.apply(testData).get()
    assert(firstResults.collect().toSeq === Seq(42 + firstOffset, 58 + firstOffset, 61 + firstOffset))

    val (secondPipeline, secondModel) = intEstimator.withData(dataTransformer(trainData), labelTransformer(trainLabels), firstModel)
    val secondOffset = firstOffset * 2
    assert(secondModel.get == secondOffset)
    val secondResults = secondPipeline.apply(testData).get()

    assert(secondResults.collect().toSeq === Seq(42 + secondOffset, 58 + secondOffset, 61 + secondOffset))
  }
}
