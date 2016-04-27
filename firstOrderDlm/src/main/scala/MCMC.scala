import breeze.stats.distributions.{Uniform, Gaussian, Gamma}
import dlm._
import KalmanFilter._
import java.io.{File, PrintWriter}
import breeze.numerics.log
import breeze.linalg.{DenseMatrix, DenseVector, diag}
import scalaz._
import Scalaz._

object MCMC {
  case class MetropolisState(params: Parameters, accepted: Int, ll: Loglikelihood)

  /**
    * Adds gaussian noise to a double, but rejects if the number is below lower
    */
  def gaussianNoiseBound(lower: Double, noise: Double): Double => Double = {
    x => {
      val x1 = x + Gaussian(0, noise).draw
      if (x1 < lower) gaussianNoiseBound(lower, noise)(x)
      else x1
  }}

    /**
      * Moves the parameters according to a random walk
      */
  def perturb(delta: Double): Parameters => Parameters = {
    p =>
    Parameters(gaussianNoiseBound(0, delta)(p.v), gaussianNoiseBound(0, delta)(p.w), p.m0, p.c0)
  }
  /**
    * A metropolis hastings step
    */
  def metropolisStep(
    likelihood: Parameters => Loglikelihood,
    perturb: Parameters => Parameters)(
    state: MetropolisState): Option[(MetropolisState, MetropolisState)] = {

    val propParam = perturb(state.params)
    val propLl = likelihood(propParam)

    if (log(Uniform(0,1).draw) < propLl - state.ll) {
      Some((state, MetropolisState(propParam, state.accepted + 1, propLl)))
    } else {
      Some((state, state))
    }
  }

  /**
    * Generate iterations using scalaz unfold
    */
  def metropolisIters(
    initParams: Parameters,
    likelihood: Parameters => Loglikelihood,
    perturb: Parameters => Parameters): Stream[MetropolisState] = {

    val initState = MetropolisState(initParams, 0, likelihood(initParams))
    unfold(initState)(metropolisStep(likelihood, perturb))
  }

  def main(args: Array[String]): Unit = {
    val n = 10000
    val p = Parameters(3.0, 0.5, 0.0, 10.0)
    val observations = simulate(p).take(100).toVector

    val iters = metropolisIters(p, filterll(observations), perturb(0.1)).take(n)
    println(s"Accepted: ${iters.last.accepted.toDouble/n}")

    // write the parameters to file
    val pw = new PrintWriter(new File("mcmcOut.csv"))
    pw.write(iters.map(_.params).mkString("\n"))
    pw.close()
  }
}
