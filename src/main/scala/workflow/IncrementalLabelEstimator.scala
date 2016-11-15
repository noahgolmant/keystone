package workflow

import org.apache.spark.rdd.RDD
import org.bouncycastle.crypto.macs.OldHMac

import scala.reflect.ClassTag

/**
  * Created by noah on 10/28/16.
  */
abstract class IncrementalLabelEstimator[A, B, L, M] extends IncrementalEstimatorOperator {

  def withData(data: RDD[A], labels: RDD[L], oldModel: M): (Pipeline[A, B], PipelineDatum[M]) = {
    withData(PipelineDataset(data), PipelineDataset(labels), PipelineDatum(oldModel))
  }

  def withData(data: RDD[A], labels: RDD[L], oldModel: PipelineDatum[M]): (Pipeline[A, B], PipelineDatum[M]) = {
    withData(PipelineDataset(data), PipelineDataset(labels), oldModel)
  }

  def withData(data: PipelineDataset[A], labels: RDD[L], oldModel: M): (Pipeline[A, B], PipelineDatum[M]) = {
    withData(data, PipelineDataset(labels), PipelineDatum(oldModel))
  }

  def withData(data: PipelineDataset[A], labels: RDD[L], oldModel: PipelineDatum[M]): (Pipeline[A, B], PipelineDatum[M]) = {
    withData(data, PipelineDataset(labels), oldModel)
  }

  def withData(data: PipelineDataset[A], labels: PipelineDataset[L], oldModel: PipelineDatum[M]): (Pipeline[A, B], PipelineDatum[M]) = {
    // Add the data input and the labels inputs into the same Graph
    val (dataLabelsGraph, _, _, labelSinkMapping) =
      data.executor.graph.addGraph(labels.executor.graph)


    val (dataLabelsModelGraph, _, _, labelModelSinkMapping) =
      dataLabelsGraph.addGraph(oldModel.executor.graph)

    // Remove the data sink & labels & old model sinks,
    // Then insert this label estimator into the graph with the data & labels as the inputs
    val dataSink = dataLabelsModelGraph.getSinkDependency(data.sink)
    val labelsSink = labels.executor.graph.getSinkDependency(labels.sink)
    val modelSink = dataLabelsModelGraph.getSinkDependency(labelModelSinkMapping(oldModel.sink))
    val (estimatorWithLabels, labelId) = dataLabelsModelGraph
      .removeSink(data.sink)
      .addNode(labels.executor.graph.getOperator(labelsSink.asInstanceOf[NodeId]), Seq())


    val (estimatorWithInputs, estId) = estimatorWithLabels
      .removeSink(labelModelSinkMapping(oldModel.sink))
      .addNode(this, Seq(dataSink, labelId, modelSink))

    // Now that the labeled estimator is attached to the data & old model, we need to build a pipeline DAG
    // that applies the fit output of the estimator. We do this by creating a new Source in the DAG,
    // Adding a delegating transformer that depends on the source and the label estimator,
    // And finally adding a sink that connects to the delegating transformer.
    val (estGraphWithNewSource, sourceId) = estimatorWithInputs.addSource()
    val (almostFinalGraph, delegatingId) = estGraphWithNewSource.addNode(new DelegatingOperator, Seq(estId, sourceId))
    val (newGraph, sinkId) = almostFinalGraph.addSink(delegatingId)

    // Finally, we construct a new pipeline w/ the new graph & new state.
    val newModel = fit(data.get, labels.get, oldModel.get)

    (new Pipeline(new GraphExecutor(newGraph), sourceId, sinkId), PipelineDatum(newModel))
  }

  override def fitRDDs(inputs: Seq[DatasetExpression], oldModel: DatumExpression): TransformerOperator = {
    val data = inputs.head.get.asInstanceOf[RDD[A]]
    val oldModelDatum = oldModel.get.asInstanceOf[M]
    val model = fit(data, inputs(1).get.asInstanceOf[RDD[L]], oldModelDatum)
    transformer(model)
  }

  def initialModel(): M

  def fit(data: RDD[A], labels: RDD[L], oldModel: M): M

  def transformer(model: M): Transformer[A, B]

  def getInitialModel(): PipelineDatum[M] = PipelineDatum(initialModel())

}
