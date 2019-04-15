import breeze.stats.distributions.{Uniform, Gaussian, Gamma}
import dlm._
import KalmanFilter._
import java.io.{File, PrintWriter}
import math.{log, exp}
import breeze.linalg.{DenseMatrix, DenseVector, diag}

object MCMC {
  case class MetropolisState(params: Parameters, accepted: Int, ll: Loglikelihood)

  /**
    * Moves the parameters according to a random walk
    */
  def perturb(delta: (Double, Double)): Parameters => Parameters = p => {
    Parameters(
      p.v * exp(Gaussian(0, delta._1).draw),
      p.w * exp(Gaussian(0, delta._2).draw),
      p.m0, p.c0)
  }

  /**
    * A metropolis step
    */
  def metropolisStep(
    likelihood: Parameters => Loglikelihood,
    perturb: Parameters => Parameters): MetropolisState => MetropolisState = state => {

    val propParam = perturb(state.params)
    val propLl = likelihood(propParam)

    if (log(Uniform(0,1).draw) < propLl - state.ll) {
      MetropolisState(propParam, state.accepted + 1, propLl)
    } else {
      state
    }
  }

  /**
    * Generate iterations using Stream.iterate
    */
  def metropolisIters(
    initParams: Parameters,
    likelihood: Parameters => Loglikelihood,
    perturb: Parameters => Parameters): Stream[MetropolisState] = {

    val initState = MetropolisState(initParams, 0, likelihood(initParams))

    Stream.iterate(initState)(metropolisStep(likelihood, perturb))
  }

  def main(args: Array[String]): Unit = {
    val n = 10000
    val p = Parameters(3.0, 0.5, 0.0, 10.0)
    val observations = simulate(p).take(100).toVector

    val iters = metropolisIters(p, filterll(observations), perturb((1.0, 0.1))).take(n)
    println(s"Accepted: ${iters.last.accepted.toDouble/n}")

    // write the parameters to file
    val pw = new PrintWriter(new File("mcmcOut.csv"))
    pw.write(iters.map(_.params).mkString("\n"))
    pw.close()
  }
}
