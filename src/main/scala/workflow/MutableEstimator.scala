package workflow

import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
  * An estimator has a `fitRDD` method which takes an input and emits a [[Transformer]]
  * @tparam A The type of input this estimator (and the resulting Transformer) takes
  * @tparam B The output type of the Transformer this estimator produces when being fit
  */
abstract class MutableEstimator[A, B : ClassTag] extends EstimatorOperator {
  /**
    * Constructs a pipeline that fits this estimator to training data,
    * then applies the resultant transformer to the Pipeline input.
    *
    * @param data The training data
    * @return A pipeline that fits this estimator and applies the result to inputs.
    */
  final def withData(data: RDD[A]): Pipeline[A, B] = {
    withData(PipelineDataset(data))
  }

  /**
    * Keep track of the mutable transformer created by this estimator so we can update it given new data.
    */
  protected var transformer: Option[MutableTransformer[A, B]] = None
  protected var currentData: Option[PipelineDataset[A]] = None

  /**
    * Constructs a pipeline that fits this estimator to training data,
    * then applies the resultant transformer to the Pipeline input.
    *
    * @param data The training data
    * @return A pipeline that fits this estimator and applies the result to inputs.
    */
  final def withData(data: PipelineDataset[A]): Pipeline[A, B] = {
    // Remove the data sink,
    // Then insert this estrimator into the graph with the data as the input
    val curSink = data.executor.graph.getSinkDependency(data.sink)
    val (estGraph, estId) = data.executor.graph.removeSink(data.sink).addNode(this, Seq(curSink))

    // Now that the estimator is attached to the data, we need to build a pipeline DAG
    // that applies the fit output of the estimator. We do this by creating a new Source in the DAG,
    val (estGraphWithNewSource, sourceId) = estGraph.addSource()

    // Adding a delegating transformer that depends on the source and the label estimator,
    val (almostFinalGraph, delegatingId) = estGraphWithNewSource.addNode(new DelegatingOperator, Seq(estId, sourceId))

    // And finally adding a sink that connects to the delegating transformer.
    val (newGraph, sinkId) = almostFinalGraph.addSink(delegatingId)

    // Store the initial data used to create the transformer
    currentData = Some(data)

    new Pipeline(new GraphExecutor(newGraph), sourceId, sinkId)
  }


  /**
    * The non-type-safe `fitRDDs` method of [[EstimatorOperator]] that is being overridden by the Estimator API.
    */
  final override private[workflow] def fitRDDs(inputs: Seq[DatasetExpression]): TransformerOperator = {
    transformer = Some(new MutableTransformer(fit(inputs.head.get.asInstanceOf[RDD[A]])))
    transformer.get
  }


  /**
    * Add new data to re-train this estimator's transformer. Updates the current data
    * and replaces the current [[MutableTransformer]]'s [[Transformer]] state with a new one.
    * @param newData New training data to add to the estimator.
    */
  def addData(newData: RDD[A]) = {
    currentData = currentData match {
      case Some(data) => Some(PipelineDataset(data.get().union(newData)))
      case None => Some(PipelineDataset(newData))
    }

    val newTransformer = fit(currentData.get.get())
    transformer.get.update(newTransformer)
  }

  /**
    * The type-safe method that ML developers need to implement when writing new Estimators.
    *
    * @param data The estimator's training data.
    * @return A new transformer
    */
  def fit(data: RDD[A]): Transformer[A, B]
}
