package workflow

import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
  * Transformers are operators that may be applied both to single input items and to RDDs of input items.
  * They may be chained together, along with [[Estimator]]s and [[LabelEstimator]]s, to produce complex
  * pipelines.
  *
  * Transformer extends [[Pipeline]], meaning that its publicly exposed methods for transforming data
  * and chaining are implemented there.
  *
  * @tparam A input item type the transformer takes
  * @tparam B output item type the transformer produces
  */
private[workflow] class MutableTransformer[A, B : ClassTag](initialState: Transformer[A, B]) extends TransformerOperator with Chainable[A, B] {

  private var currentState: Transformer[A, B] = initialState

  override def toPipeline: Pipeline[A, B] = new Pipeline(
    executor = new GraphExecutor(Graph(
      sources = Set(SourceId(0)),
      sinkDependencies = Map(SinkId(0) -> NodeId(0)),
      operators = Map(NodeId(0) -> this),
      dependencies = Map(NodeId(0) -> Seq(SourceId(0)))
    )),
    source = SourceId(0),
    sink = SinkId(0)
  )

  /**
    * The application of this Transformer to a single input item.
    * This method MUST be overridden by ML developers.
    *
    * @param in  The input item to pass into this transformer
    * @return  The output value
    */
  final def apply(in: A): B = currentState(in)


  def update(newState: Transformer[A, B]) = {
    currentState = newState
  }

  /**
    * The application of this Transformer to an RDD of input items.
    * This method may optionally be overridden by ML developers.
    *
    * @param in The bulk RDD input to pass into this transformer
    * @return The bulk RDD output for the given input
    */
  final def apply(in: RDD[A]): RDD[B] = currentState(in)

  final override private[workflow] def singleTransform(inputs: Seq[DatumExpression]): Any = {
    apply(inputs.head.get.asInstanceOf[A])
  }

  final override private[workflow] def batchTransform(inputs: Seq[DatasetExpression]): RDD[_] = {
    apply(inputs.head.get.asInstanceOf[RDD[A]])
  }
}
