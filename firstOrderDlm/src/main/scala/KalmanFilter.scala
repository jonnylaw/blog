import breeze.stats.distributions.Gaussian
import dlm._
import java.io.{PrintWriter, File}

object KalmanFilter {

  case class FilterOut(data: Data, p: Parameters, likelihood: Loglikelihood) {
    override def toString = data.toString + ", " + p.toString
  }

  def filterll(data: Seq[Data])(params: Parameters): Loglikelihood = {
    val initFilter = FilterOut(data.head, params, 0.0) // initialise the filter

    data.tail.foldLeft(initFilter)(filter).likelihood
  }

  def filter(d: FilterOut, y: Data): FilterOut = {
    // update the mean and variance of the posterior to determine the state space
    val e1 = y.observation - d.p.m0
    val a1 = (d.p.c0 + d.p.w)/(d.p.c0 + d.p.w + d.p.v)
    val m1 = d.p.m0 + a1 * e1
    val c1 = a1 * d.p.v

    val likelihood = Gaussian(d.p.m0, d.p.c0 + d.p.w + d.p.v).logPdf(y.observation)

    // return the data with the expectation of the hidden state and the updated Parameters
    FilterOut(Data(y.time, y.observation, Some(m1)), Parameters(d.p.v, d.p.w, m1, c1), likelihood)
  }

  def filterSeries(data: Seq[Data])(params: Parameters): Seq[FilterOut] = {
    val initFilter = FilterOut(data.head, params, 0.0) // initialise the filter

    data.tail.scanLeft(initFilter)(filter)
  }

  val runKalmanFilter = {
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
