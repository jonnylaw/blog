package model

import breeze.stats.distributions._
import breeze.linalg._
import breeze.numerics.exp
import java.io.{File, PrintWriter}

object GaussianModel {
  case class Parameters(mu: DenseVector[Double], sigma: Double) {
    override def toString = s"${mu.data.mkString(", ")}, $sigma"
  }

  /**
    * Specify a model which simulates pairs of independent gaussian random variables
    */
  def model(params: Parameters) = 
    MultivariateGaussian(params.mu, diag(DenseVector.fill(2)(params.sigma)))

  def likelihood(points: Seq[DenseVector[Double]])(p: Parameters) =
    points.map { point => 
      MultivariateGaussian(p.mu, diag(DenseVector.fill(2)(p.sigma))).logPdf(point)
    }.reduce((x, y) => x + y)

  def prior(p: Parameters) = {
    MultivariateGaussian(DenseVector(2.0, 3.0), diag(DenseVector.fill(2)(3.0))).logPdf(p.mu) +
      Gamma(shape = 0.5, scale = 2.0).logPdf(1/p.sigma)
  }

  def simPrior = for {
    mu <- MultivariateGaussian(DenseVector(2.0, 3.0), diag(DenseVector.fill(2)(3.0)))
    sigma <- Gamma(2.0, 3.0)
  } yield Parameters(mu, sigma)

  def propose(scale: Double)(p: Parameters) = 
    for {
      innov <- MultivariateGaussian(DenseVector.fill(3)(0.0), diag(DenseVector.fill(3)(scale)))
      mu = p.mu + innov(0 to 1)
      sigma = p.sigma * exp(innov(2))
    } yield Parameters(mu, sigma)

  def main(args: Array[String]) = {
    val p = Parameters(DenseVector(2.0, 3.0), 0.5)
    val data = model(p).sample(100)

    val pw1 = new PrintWriter(new File("data/BivariateSimulated.csv"))
    data.
      foreach( d =>
        pw1.write(s"${d(0)}, ${d(1)}\n")
      )
    pw1.close()

    def logMeasure = (p: Parameters) => likelihood(data)(p) + prior(p)

    val params = MarkovChain.metropolis(simPrior.draw, propose(0.025))(logMeasure).
      steps

    val pw = new PrintWriter(new File("data/Parameters.csv"))
    params.
      take(10000).
      foreach { p =>
        pw.write(p.toString + "\n")
      }
    pw.close()
  }
}
