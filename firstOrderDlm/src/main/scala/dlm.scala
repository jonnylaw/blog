import java.io.{File, PrintWriter}
import breeze.stats.distributions._
import Math.sqrt
import KalmanFilter._

object dlm {
  type Loglikelihood = Double
  type Observation = Double
  type State = Double
  type Time = Double

  case class Data(time: Time, observation: Observation, state: Option[State]) {
    override def toString = state match {
      case Some(x) => s"$time, $observation, $x"
      case None => s"$time, $observation"
    }
  }

  case class Parameters(v: Double, w: Double, m0: Double, c0: Double) {
    override def toString = s"$v, $w, $m0, $c0" 
  }

  def simulate(p: Parameters): Stream[Data] = {
    val stateSpace = Stream.iterate(Gaussian(p.m0, sqrt(p.c0)).draw)(x =>
      x + Gaussian(0, sqrt(p.w)).draw)

    stateSpace.zipWithIndex map { case (x, t) => 
      Data(t, x + Gaussian(0, sqrt(p.v)).draw, Some(x)) }
  }

  def step_rw(p: Parameters): Double => Rand[Double] = 
    x => Gaussian(x, p.v)

  def sim_markov(p: Parameters, init: Double): Process[Double] = {
    MarkovChain(init)(step_rw(p))
  }

  def main(args: Array[String]): Unit = {

    val p = Parameters(3.0, 0.5, 0.0, 10.0)

    // simulate 16 different realisations of 100 observations, representing 16 stations
    val observations = (1 to 16) map (id => (id, simulate(p).take(100).toVector))

    val pw = new PrintWriter(new File("firstOrderDlm.csv"))
    pw.write(
      observations.
        flatMap{ case (id, data) =>
          data map (x => id + ", " + x.toString)}.
        mkString("\n"))
    pw.close()
  }
}
