package nodes.learning.incremental


import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD, LogisticRegressionModel => MLlibLRM}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLlibVector}
import breeze.linalg.Vector
import org.apache.spark.mllib.optimization.{GradientDescent, LBFGS, LogisticGradient, SquaredL2Updater}
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.mllib.util.DataValidators
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

  /**
    * Train a logistic regression model using mini-batch SGD.
    * @param stepSize
    * @param numIterations
    * @param regParam
    * @param miniBatchFraction
    */
  private[this] class LogisticRegressionWithSGD (
      private var stepSize: Double,
      private var numIterations: Int,
      private var regParam: Double,
      private var miniBatchFraction: Double)
    extends GeneralizedLinearAlgorithm[MLlibLRM] with Serializable {

    private val gradient = new LogisticGradient()
    private val updater = new SquaredL2Updater()

    override val optimizer = new GradientDescent(gradient, updater)
      .setStepSize(stepSize)
      .setNumIterations(numIterations)
      .setRegParam(regParam)
      .setMiniBatchFraction(miniBatchFraction)
    override protected val validators = List(DataValidators.binaryLabelValidator)

    override protected[mllib] def createModel(weights: MLlibVector, intercept: Double) = {
      new MLlibLRM(weights, intercept)
    }

  }


  override def fit(data: RDD[T], labels: RDD[Int], oldModel: MLlibLRM = initialModel): MLlibLRM = {
    val labeledPoints = labels.zip(data).map(x => LabeledPoint(x._1, breezeVectorToMLlib(x._2)))
    val trainer = new LogisticRegressionWithSGD(
      stepSize,
      numIters,
      regParam,
      miniBatchFraction)

    trainer.setValidateData(false)

    val model = trainer.run(labeledPoints, oldModel.weights)
    model
  }

  override def transformer(model: MLlibLRM): Transformer[T, Double] = new Transformer[T, Double] {
    override def apply(in: T): Double = model.predict(breezeVectorToMLlib(in))
  }
}
