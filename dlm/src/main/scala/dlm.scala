import scalaz._
import Scalaz._
import java.io.{File, PrintWriter}
import breeze.stats.distributions.Gaussian
import Math.sqrt

object dlm {
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
    val stateSpace = unfold(Gaussian(p.m0, sqrt(p.c0)).draw)(x =>
      Some(x, x + Gaussian(0, sqrt(p.w)).draw)
    )
    stateSpace.zipWithIndex map { case (x, t) => 
      Data(t, x + Gaussian(0, sqrt(p.v)).draw, Some(x)) }
  }

  def main(args: Array[String]) = {
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
