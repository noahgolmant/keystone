package nodes.learning.incremental


import org.apache.spark.mllib.classification.{LogisticRegressionModel => MLlibLRM}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLlibVector}
import breeze.linalg.Vector
import org.apache.spark.mllib.optimization.{GradientDescent, LogisticGradient, SquaredL2Updater}
import utils.MLlibUtils.breezeVectorToMLlib
import workflow.{IncrementalLabelEstimator, Transformer}

import scala.reflect.ClassTag


/**
  * Created by noah on 10/31/16.
  */
class IncrementalLogisticRegressionEstimator[T <: Vector[Double] : ClassTag](
        numClasses: Int,
        stepSize: Double,
        miniBatchFraction: Double,
        regParam: Double = 0,
        numIters: Int = 100,
        convergenceTol: Double = 1E-4,
        numFeatures: Int = -1
      ) extends IncrementalLabelEstimator[T, Double, Int, MLlibLRM] {

  private val numOfLinearPredictor = numClasses - 1
  private val initialWeights = {
    if (numOfLinearPredictor == 1) {
      Vectors.zeros(numFeatures)
    } else {
      Vectors.zeros(numFeatures * numOfLinearPredictor)
    }
  }
  def initialModel(): MLlibLRM = {
    if (numOfLinearPredictor == 1) {
      new MLlibLRM(initialWeights, 0.0)
    } else {
      new MLlibLRM(initialWeights, 0.0, numFeatures, numOfLinearPredictor + 1)
    }
  }

  override def fit(data: RDD[T], labels: RDD[Int], oldModel: MLlibLRM = initialModel): MLlibLRM = {
    //val labeledPoints = labels.zip(data).map(x => LabeledPoint(x._1, breezeVectorToMLlib(x._2)))
    val labeledPoints = labels.zip(data).map(x => (x._1.toDouble, breezeVectorToMLlib(x._2)))
    val (weights, _) = GradientDescent.runMiniBatchSGD(
      labeledPoints,
      new LogisticGradient(),
      new SquaredL2Updater(),
      stepSize,
      numIters,
      regParam,
      miniBatchFraction,
      initialWeights,
      convergenceTol
    )

    val model = new MLlibLRM(weights, 0)
    model
  }

  override def transformer(model: MLlibLRM): Transformer[T, Double] = new Transformer[T, Double] {
    override def apply(in: T): Double = model.predict(breezeVectorToMLlib(in))
  }
}
