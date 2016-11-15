package workflow

import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
  * Created by noah on 10/28/16.
  */
abstract class IncrementalEstimator[A, B, M] extends IncrementalEstimatorOperator {

  def withData(data: RDD[A], oldModel: M): (Pipeline[A, B], PipelineDatum[M]) = withData(PipelineDataset(data), PipelineDatum(oldModel))

  def withData(data: RDD[A], oldModel: PipelineDatum[M]): (Pipeline[A, B], PipelineDatum[M]) = withData(PipelineDataset(data), oldModel)

  def withData(data: PipelineDataset[A], oldModel: PipelineDatum[M]): (Pipeline[A, B], PipelineDatum[M]) = {
    // Add the data input and the labels inputs into the same Graph
    val (dataAndModel, _, _, modelSinkMapping) =
      data.executor.graph.addGraph(oldModel.executor.graph)

    // Remove the data sink & the oldModel sink,
    // Then insert this label estimator into the graph with the data & labels as the inputs
    val dataSink = dataAndModel.getSinkDependency(data.sink)
    val labelsSink = dataAndModel.getSinkDependency(modelSinkMapping(oldModel.sink))
    val (estimatorWithInputs, estId) = dataAndModel
      .removeSink(data.sink)
      .removeSink(modelSinkMapping(oldModel.sink))
      .addNode(this, Seq(dataSink, labelsSink))

    // Now that the labeled estimator is attached to the data & old model, we need to build a pipeline DAG
    // that applies the fit output of the estimator. We do this by creating a new Source in the DAG,
    // Adding a delegating transformer that depends on the source and the label estimator,
    // And finally adding a sink that connects to the delegating transformer.
    val (estGraphWithNewSource, sourceId) = estimatorWithInputs.addSource()
    val (almostFinalGraph, delegatingId) = estGraphWithNewSource.addNode(new DelegatingOperator, Seq(estId, sourceId))
    val (newGraph, sinkId) = almostFinalGraph.addSink(delegatingId)

    // Finally, we construct a new pipeline w/ the new graph & new state.
    val newModel = fit(data.get, oldModel.get)

    (new Pipeline(new GraphExecutor(newGraph), sourceId, sinkId), PipelineDatum(newModel))
  }

  override def fitRDDs(inputs: Seq[DatasetExpression], oldModel: DatumExpression): TransformerOperator = {
    val data = inputs.head.get.asInstanceOf[RDD[A]]
    val oldModelDatum = oldModel.get.asInstanceOf[M]
    val model = fit(data, oldModelDatum)
    transformer(model)
  }

  def fit(data: RDD[A], oldModel: M): M

  def transformer(model: M): Transformer[A, B]
}
