// import breeze.linalg._
// import breeze.stats.distributions._
// import breeze.numerics._
// import java.io.PrintWriter

// case class KalmanFilterState(time: Double, mean: DenseVector[Double], covariance: DenseMatrix[Double], ll: Double)

// case class KalmanFilter(model: Dlm) {
//   def init: Parameters => KalmanFilterState = p => KalmanFilterState(0.0, p.m0, cholesky(p.c0), 0.0)

//   def filterStep(p: Parameters): (KalmanFilterState, Data) => KalmanFilterState = (s, d) => {
//     val a = model.g * s.mean
//     val r = model.g * s.covariance * model.g.t + p.w

//     val f = model.f * a
//     val q = model.f * r * r.t * model.f.t + p.v
//     val e = d.observation - f

//     // kalman gain
//     val k = (r * r.t * model.f.t) * inv(q)

//     // compute joseph form update to covariance
//     val i = DenseMatrix.eye[Double](s.covariance.cols)
//     val covariance = (i - k * model.f) * r * (i - k * model.f).t + k * p.v * k.t

//     // update the marginal likelihood
//     val ll = s.ll + Gaussian(f, q).logPdf(d.observation)

//     KalmanFilterState(d.time, a + k * e, covariance, ll)
//   }

//   def filter(p: Parameters, d: Seq[Data]): Seq[KalmanFilterState] = {
//     d.scanLeft(init)(filterStep(p))
//   }
// }
