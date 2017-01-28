package nodes.learning.incremental


import org.apache.spark.mllib.classification.{LogisticRegressionModel => MLlibLRM}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLlibVector}
import breeze.linalg.Vector
import org.apache.spark.mllib.optimization.{GradientDescent, LBFGS, LogisticGradient, SquaredL2Updater}
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.mllib.util.DataValidators
import utils.MLlibUtils.breezeVectorToMLlib
import workflow.{IncrementalLabelEstimator, Transformer}

import scala.reflect.ClassTag


case class Model(lrModel: MLlibLRM, iteration: Float)

/**
  * Created by noah on 10/31/16.
  */
class ModelAvgIncrementaLREstimator[T <: Vector[Double] : ClassTag](
                                                                              numClasses: Int,
                                                                              stepSize: Double,
                                                                              miniBatchFraction: Double,
                                                                              regParam: Double = 0,
                                                                              numIters: Int = 100,
                                                                              convergenceTol: Double = 1E-4,
                                                                              numFeatures: Int = -1
                                                                            ) extends IncrementalLabelEstimator[T, Double, Int, Model] {

  private val numOfLinearPredictor = numClasses - 1
  private val initialWeights = {
    if (numOfLinearPredictor == 1) {
      Vectors.zeros(numFeatures)
    } else {
      Vectors.zeros(numFeatures * numOfLinearPredictor)
    }
  }
  def initialModel(): Model = {
    if (numOfLinearPredictor == 1) {
      new Model(new MLlibLRM(initialWeights, 0.0), 1)
    } else {
      new Model(new MLlibLRM(initialWeights, 0.0, numFeatures, numOfLinearPredictor + 1), 1)
    }
  }

  /**
    * Train a classification model for Multinomial/Binary Logistic Regression using
    * Limited-memory BFGS. Standard feature scaling and L2 regularization are used by default.
    * NOTE: Labels used in Logistic Regression should be {0, 1, ..., k - 1}
    * for k classes multi-label classification problem.
    */
  private[this] class LogisticRegressionWithLBFGS(numClasses: Int, numFeaturesValue: Int)
    extends GeneralizedLinearAlgorithm[MLlibLRM] with Serializable {

    this.numFeatures = numFeaturesValue
    override val optimizer = new LBFGS(new LogisticGradient, new SquaredL2Updater)

    override protected val validators = List(multiLabelValidator)

    require(numClasses > 1)
    numOfLinearPredictor = numClasses - 1
    if (numClasses > 2) {
      optimizer.setGradient(new LogisticGradient(numClasses))
    }

    private def multiLabelValidator: RDD[LabeledPoint] => Boolean = { data =>
      if (numOfLinearPredictor > 1) {
        DataValidators.multiLabelValidator(numOfLinearPredictor + 1)(data)
      } else {
        DataValidators.binaryLabelValidator(data)
      }
    }

    override protected def createModel(weights: MLlibVector, intercept: Double) = {
      if (numOfLinearPredictor == 1) {
        new MLlibLRM(weights, intercept)
      } else {
        new MLlibLRM(weights, intercept, numFeatures, numOfLinearPredictor + 1)
      }
    }
  }

  override def fit(data: RDD[T], labels: RDD[Int], oldModel: Model = initialModel): Model = {
    val labeledPoints = labels.zip(data).map(x => LabeledPoint(x._1, breezeVectorToMLlib(x._2)))
    //val labeledPoints = labels.zip(data).map(x => (x._1.toDouble, breezeVectorToMLlib(x._2)))

    val trainer = new LogisticRegressionWithLBFGS(numClasses, numFeatures)
    trainer.setValidateData(false).optimizer.setNumIterations(numIters).setRegParam(regParam)
    val model = trainer.run(labeledPoints)
    val weights: Vector[Double] = Vector(model.weights.toArray)

    val newCount = oldModel.iteration + 1
    val oldWeightVector = Vector(oldModel.lrModel.weights.toArray)
    val cumulativeWeights: MLlibVector = Vectors.dense(((weights + (oldWeightVector :* oldModel.iteration.toDouble)) :* (1.0 / newCount)).toArray)

    val lrModel = new MLlibLRM(cumulativeWeights, 0.0)
    new Model(lrModel, newCount)
  }

  override def transformer(model: Model): Transformer[T, Double] = new Transformer[T, Double] {
    override def apply(in: T): Double = model.lrModel.predict(breezeVectorToMLlib(in))
  }
}
