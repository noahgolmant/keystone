package nodes.nlp

import pipelines.Transformer

import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
 * An ngram featurizer.
 *
 * @param orders valid ngram orders, must be consecutive positive integers
 */
class NGramsFeaturizer(orders: Seq[Int]) extends Transformer[Seq[String], Seq[Seq[String]]] {

  private[this] final val minOrder = orders.min
  private[this] final val maxOrder = orders.max

  require(minOrder >= 1, s"minimum order is not >= 1, found $minOrder")
  orders.sliding(2).foreach {
    case xs if xs.length > 1 => require(xs(0) == xs(1) - 1,
      s"orders are not consecutive; contains ${xs(0)} and ${xs(1)}")
    case _ =>
  }

  override def apply(in: Seq[String]): Seq[Seq[String]] = {
    val ngramBuf = new ArrayBuffer[String](orders.max)
    var j = 0
    var order = 0
    val ngramsBuf = new ArrayBuffer[Seq[String]]()
    var i = 0
    while (i + minOrder <= in.length) {
      ngramBuf.clear()

      j = i
      while (j < i + minOrder) {
        ngramBuf += in(j)
        j += 1
      }
      ngramsBuf += ngramBuf.clone()

      order = minOrder + 1
      while (order <= maxOrder && i + order <= in.length) {
        ngramBuf += in(i + order - 1)
        ngramsBuf += ngramBuf.clone()
        order += 1
      }
      i += 1
    }
    ngramsBuf
  }
}