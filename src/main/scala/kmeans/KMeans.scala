package kmeans

import scala.annotation.tailrec
import scala.collection.{Map, Seq, mutable}
import scala.collection.parallel.CollectionConverters.*
import scala.collection.parallel.{ForkJoinTaskSupport, ParMap, ParSeq}
import scala.util.Random
import org.scalameter.*
import java.util.concurrent.ForkJoinPool

class KMeans extends KMeansInterface:

  def generatePoints(k: Int, num: Int): ParSeq[Point] =
    val randx = Random(1)
    val randy = Random(3)
    val randz = Random(5)
    (0 until num)
      .map({ i =>
        val x = ((i + 1) % k) * 1.0 / k + randx.nextDouble() * 0.5
        val y = ((i + 5) % k) * 1.0 / k + randy.nextDouble() * 0.5
        val z = ((i + 7) % k) * 1.0 / k + randz.nextDouble() * 0.5
        Point(x, y, z)
      }).to(mutable.ArrayBuffer).par

  def initializeMeans(k: Int, points: ParSeq[Point]): ParSeq[Point] =
    val rand = Random(7)
    (0 until k).map(_ => points(rand.nextInt(points.length))).to(mutable.ArrayBuffer).par


  def findClosest(p: Point, means: IterableOnce[Point]): Point =
    val it = means.iterator
    assert(it.nonEmpty)
    var closest = it.next()
    var minDistance = p.squareDistance(closest)
    while it.hasNext do
      val point = it.next()
      val distance = p.squareDistance(point)
      if distance < minDistance then
        minDistance = distance
        closest = point
    closest


  /*
  These two methods take a sequence of points and a sequence of means, and return
  a map collection, which maps each mean to the sequence of points
  in the corresponding cluster.
  Hint: Use groupBy and the findClosest method, which is already defined for you.
  at all the means are in the resulting map, even if their sequences are empty.
  Map<the most near point, [its asscoiated points]>
  */
  def classify(points: ParSeq[Point], means: ParSeq[Point]): ParMap[Point, ParSeq[Point]] =
      points.groupBy(findClosest(_, means)).toMap

  def findAverage(oldMean: Point, points: ParSeq[Point]): Point = if points.isEmpty then oldMean else
    var x = 0.0
    var y = 0.0
    var z = 0.0
    points.seq.foreach { p =>
      x += p.x
      y += p.y
      z += p.z
    }
    Point(x / points.length, y / points.length, z / points.length)


  //for each current median, we cacculate  its asscioated points's new median
  def update(classified: ParMap[Point, ParSeq[Point]], oldMeans: ParSeq[Point]): ParSeq[Point] =
    oldMeans.map(m => findAverage(m, classified(m)))


  /*Finally, you will implement convergence detection. The convergence detection method takes
  a sequence of old means and the sequence of updated means, and returns a boolean
  indicating if the algorithm converged or not. Given an eta parameter,
  oldMeans and newMeans, it returns true if the algorithm converged,
  and false otherwise:
  The algorithm converged iff the square distance between the old and the new mean is less than or equal to eta, for all means.

Note: the means in the two lists are ordered -- the mean at i in oldMeans is the previous value of the mean at i in newMeans.

   */

  def converged(eta: Double, oldMeans: ParSeq[Point], newMeans: ParSeq[Point]): Boolean =
    val t:ParSeq[Double] = oldMeans.zip(newMeans).map( x => x._1.squareDistance(x._2)).filter(_ > eta)
    t.isEmpty

  @tailrec
  final def kMeans(points: ParSeq[Point], means: ParSeq[Point], eta: Double): ParSeq[Point] =
    // your implementation need to be tail recursive
    val newMeans = update(classify(points, means), means)
    if (!converged(eta,means, newMeans)) kMeans(points, newMeans, eta) else newMeans

/** Describes one point in three-dimensional space.
 *
 *  Note: deliberately uses reference equality.
 */
class Point(val x: Double, val y: Double, val z: Double):
  private def square(v: Double): Double = v * v
  def squareDistance(that: Point): Double =
    square(that.x - x)  + square(that.y - y) + square(that.z - z)
  private def round(v: Double): Double = (v * 100).toInt / 100.0
  override def toString = s"(${round(x)}, ${round(y)}, ${round(z)})"


object KMeansRunner:

  val standardConfig = config(
    Key.exec.minWarmupRuns := 20,
    Key.exec.maxWarmupRuns := 40,
    Key.exec.benchRuns := 25,
    Key.verbose := false
  ) withWarmer(Warmer.Default())

  def main(args: Array[String]): Unit =
    val kMeans = KMeans()

    val numPoints = 500000
    val eta = 0.01
    val k = 32
    val points = kMeans.generatePoints(k, numPoints)
    val means = kMeans.initializeMeans(k, points)

    val seqtime =
      // Retrieve the support created to run the algorithm in parallel
      val parTasksupport = points.tasksupport
      // Create a support with only one executor to run the algorithm sequentially
      val seqPool = ForkJoinPool(1)
      val seqTasksupport = ForkJoinTaskSupport(seqPool)
      try
        // Run the the algorithm on the sequential support
        points.tasksupport = seqTasksupport
        means.tasksupport = seqTasksupport
        // Measure performances on the sequential runner
        standardConfig measure {
          kMeans.kMeans(points, means, eta)
        }
      finally
        // Restore the parallel support
        points.tasksupport = parTasksupport
        means.tasksupport = parTasksupport
        // Stop the sequential runner
        seqPool.shutdown()

    // Measure performances on the parallel runner
    val partime = standardConfig measure {
      kMeans.kMeans(points, means, eta)
    }

    println(s"sequential time: $seqtime")
    println(s"parallel time: $partime")
    println(s"speedup: ${seqtime.value / partime.value}")
