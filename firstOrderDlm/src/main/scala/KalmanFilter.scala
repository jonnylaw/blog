import breeze.stats.distributions.Gaussian
import dlm._
import java.io.{PrintWriter, File}

object KalmanFilter {

  case class FilterOut(data: Data, p: Parameters, likelihood: Loglikelihood) {
    override def toString = data.toString + ", " + p.toString
  }

  def filterll(data: Seq[Data])(params: Parameters): Loglikelihood = {
    val (v, w) = (params.v, params.w) // v and w are fixed
    val initFilter = Vector[FilterOut](filter(data.head, params)) // initialise the filter

    val filtered = data.tail.foldLeft(initFilter)((acc, nextObservation) => {
      // construct the parameters from the previous step of the filter
      val p = Parameters(v, w, acc.head.p.m0, acc.head.p.c0)

      // add the filtered observation to the head of the list
      filter(nextObservation, p) +: acc
    }).reverse

    filtered.foldLeft(0.0)((acc, a) => acc + a.likelihood)
  }

  def filter(d: Data, p: Parameters): FilterOut = {
    // update the mean and variance of the posterior to determine the state space
    val e1 = d.observation - p.m0
    val a1 = (p.c0 + p.w)/(p.c0 + p.w + p.v)
    val m1 = p.m0 + a1 * e1
    val c1 = a1 * p.v

    val likelihood = Gaussian(m1, c1).logPdf(d.observation)

    // return the data with the expectation of the hidden state and the updated Parameters
    FilterOut(Data(d.time, d.observation, Some(m1)), Parameters(p.v, p.w, m1, c1), likelihood)
  }

  def filterSeries(data: Seq[Data])(params: Parameters): Seq[FilterOut] = {

    val (v, w) = (params.v, params.w) // v and w are fixed
    val initFilter = Vector[FilterOut](filter(data.head, params)) // initialise the filter

    data.tail.foldLeft(initFilter)((acc, nextObservation) => {
      // construct the parameters from the previous step of the filter
      val p = Parameters(v, w, acc.head.p.m0, acc.head.p.c0)

      // add the filtered observation to the head of the list
      filter(nextObservation, p) +: acc
    }).reverse
  }


  def main(args: Array[String]) = {
    val p = Parameters(3.0, 0.5, 0.0, 10.0)
    // simulate 16 different realisations of 100 observations, representing 16 stations
    val observations = (1 to 16) map (id => (id -> simulate(p).take(100).toVector))

    // filter for one station, using simulated data
    observations.
      filter{ case (id, _) => id == 1 }.
      flatMap{ case (_, d) => filterSeries(d)(p) }

    // or, read in data from the file we previously wrote
    val data = io.Source.fromFile("firstOrderDlm.csv").getLines.toList.
      map(l => l.split(",")).
      map(r => (r(0).toInt -> Data(r(1).toDouble, r(2).toDouble, None)))

    // filter for all stations, using data from file
    val filtered = data.
      groupBy{ case (id, _) => id }. //groups by id
      map{ case (id, idAndData) =>
        (id, idAndData map (x => x._2)) }. // changes into (id, data) pairs
      map{ case (id, data) =>
        (id, filterSeries(data.sortBy(_.time))(p)) } // apply the filter to the sorted data

    // write the filter for all stations to a file
    val pw = new PrintWriter(new File("filteredDlm.csv"))
    pw.write(filtered.
      flatMap{ case (id, data) =>
          data map (x => id + ", " + x.toString)}.
      mkString("\n"))
    pw.close()
  }
}
