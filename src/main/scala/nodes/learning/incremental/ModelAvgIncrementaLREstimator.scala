package nodes.learning.incremental


import org.apache.spark.mllib.classification.{LogisticRegressionModel => MLlibLRM}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLlibVector}
import breeze.linalg.Vector
import org.apache.spark.mllib.optimization.{GradientDescent, LogisticGradient, SquaredL2Updater}
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

  override def fit(data: RDD[T], labels: RDD[Int], oldModel: Model = initialModel): Model = {
    //val labeledPoints = labels.zip(data).map(x => LabeledPoint(x._1, breezeVectorToMLlib(x._2)))
    val labeledPoints = labels.zip(data).map(x => (x._1.toDouble, breezeVectorToMLlib(x._2)))
    val (weights: Vector[Double], _) = GradientDescent.runMiniBatchSGD(
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
