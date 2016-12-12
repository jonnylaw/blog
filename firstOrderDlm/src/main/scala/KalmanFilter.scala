import breeze.stats.distributions.Gaussian
import dlm._
import java.io.{PrintWriter, File}

object KalmanFilter {

  case class FilterState(data: Data, p: Parameters, likelihood: Loglikelihood) {
    override def toString = data.toString + ", " + p.toString
  }

  def filterll(data: Seq[Data])(params: Parameters): Loglikelihood = {
    val initFilter = FilterState(data.head, params, 0.0) // initialise the filter

    data.foldLeft(initFilter)(filter(params)).likelihood
  }

  def filter(p: Parameters): (FilterState, Data) => FilterState = (s, d) => {
    val r = s.p.c0 + p.w
    val q = r + p.v
    val e = d.observation - s.p.m0

    // kalman gain
    val k = r / q
    val c1 = k * p.v
    val m1 = s.p.m0 + k * e

    val likelihood = Gaussian(s.p.m0, s.p.c0 + s.p.w + s.p.v).logPdf(d.observation)

    // return the data with the expectation of the hidden state and the updated Parameters
    FilterState(Data(d.time, d.observation, Some(m1)), Parameters(p.v, p.w, m1, c1), likelihood)
  }

  def filterSeries(data: Seq[Data])(params: Parameters): Seq[FilterState] = {
    val initFilter = FilterState(data.head, params, 0.0) // initialise the filter

    data.scanLeft(initFilter)(filter(params)).
      drop(1)
  }

  def main(args: Array[String]): Unit = {
    val p = Parameters(3.0, 0.5, 0.0, 10.0)

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
