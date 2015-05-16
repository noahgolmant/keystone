package pipelines.images.mnist

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import evaluation.MulticlassClassifierEvaluator
import loaders.{CsvDataLoader, LabeledData}
import nodes.learning.BlockLinearMapper
import nodes.misc.ZipVectors
import nodes.stats.{LinearRectifier, PaddedFFT, RandomSignNode}
import nodes.util.{ClassLabelIndicatorsFromIntLabels, MaxClassifier}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.{SparkConf, SparkContext}
import pipelines._
import scopt.OptionParser


object MnistRandomFFT extends Serializable with Logging {
  val appName = "MnistRandomFFT"

  def run(sc: SparkContext, conf: MnistRandomFFTConfig) {
    // This is a property of the MNIST Dataset (digits 0 - 9)
    val numClasses = 10

    val randomSignSource = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(conf.seed)))

    // The number of pixels in an MNIST image (28 x 28 = 784)
    val mnistImageSize = 784

    // Because the mnistImageSize is 784, we get 512 PaddedFFT features per FFT.
    // So, calculate how many FFTs are needed per block to get the desired block size.
    val fftsPerBatch = conf.blockSize / 512
    val numFFTBatches = conf.numFFTs/fftsPerBatch

    val startTime = System.nanoTime()

    val train = LabeledData(
      CsvDataLoader(sc, conf.trainLocation, conf.numPartitions)
        // The pipeline expects 0-indexed class labels, but the labels in the file are 1-indexed
        .map(x => (x(0).toInt - 1, x(1 until x.length)))
        .cache())
    val labels = ClassLabelIndicatorsFromIntLabels(numClasses).apply(train.labels)

    val batchFeaturizer = (0 until numFFTBatches).map { batch =>
      (0 until fftsPerBatch).map { x =>
        RandomSignNode(mnistImageSize, randomSignSource) then PaddedFFT then LinearRectifier(0.0)
      }
    }

    val trainingBatches = batchFeaturizer.map { x =>
      ZipVectors(x.map(y => y.apply(train.data))).cache()
    }

    // Train the model
    val blockLinearMapper = BlockLinearMapper.trainWithL2(trainingBatches, labels, conf.lambda.getOrElse(0), 1)

    val test = LabeledData(
      CsvDataLoader(sc, conf.testLocation, conf.numPartitions)
        // The pipeline expects 0-indexed class labels, but the labels in the file are 1-indexed
        .map(x => (x(0).toInt - 1, x(1 until x.length)))
        .cache())
    val actual = test.labels

    val testBatches = batchFeaturizer.toIterator.map { x =>
      ZipVectors(x.map(y => y.apply(test.data))).cache()
    }

    // Calculate train error
    blockLinearMapper.applyAndEvaluate(trainingBatches, trainPredictedValues => {
      val predicted = MaxClassifier(trainPredictedValues)
      val evaluator = MulticlassClassifierEvaluator(predicted, train.labels, numClasses)
      logInfo("Train Error is " + (100 * evaluator.totalError) + "%")
    })

    // Calculate test error
    blockLinearMapper.applyAndEvaluate(testBatches, testPredictedValues => {
      val predicted = MaxClassifier(testPredictedValues)
      val evaluator = MulticlassClassifierEvaluator(predicted, actual, numClasses)
      logInfo("TEST Error is " + (100 * evaluator.totalError) + "%")
    })

    val endTime = System.nanoTime()
    logInfo(s"Pipeline took ${(endTime - startTime)/1e9} s")
  }

  case class MnistRandomFFTConfig(
      trainLocation: String = "",
      testLocation: String = "",
      numFFTs: Int = 200,
      blockSize: Int = 4096,
      numPartitions: Int = 10,
      lambda: Option[Double] = None,
      seed: Long = 0)

  def parse(args: Array[String]): MnistRandomFFTConfig = new OptionParser[MnistRandomFFTConfig](appName) {
    head(appName, "0.1")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[Int]("numFFTs") action { (x,c) => c.copy(numFFTs=x) }
    opt[Int]("blockSize") validate { x =>
      // Bitwise trick to test if x is a power of 2
      if (((x & -x) == x) && (x >= 512)) {
        success
      } else  {
        failure("Option --blockSize must be a power of 2, and >= 512")
      }
    } action { (x,c) => c.copy(blockSize=x) }
    opt[Int]("numPartitions") action { (x,c) => c.copy(numPartitions=x) }
    opt[Double]("lambda") action { (x,c) => c.copy(lambda=Some(x)) }
    opt[Long]("seed") action { (x,c) => c.copy(seed=x) }
  }.parse(args, MnistRandomFFTConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[2]")

    val sc = new SparkContext(conf)

    val appConfig = parse(args)
    run(sc, appConfig)

    sc.stop()
  }
}
