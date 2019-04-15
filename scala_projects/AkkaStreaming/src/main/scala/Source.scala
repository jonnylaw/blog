import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import akka.stream.scaladsl._
import scala.concurrent.duration._

object Streaming {
  implicit val system = ActorSystem("Streaming")
  implicit val executor = system.dispatcher
  implicit val materializer = ActorMaterializer()

  val in = Source.tick(1.second, 1.second, 1)
  val double_flow = Flow[Int].map(a => a * 2)
  val print_sink = Sink.foreach(println)

  def main(args: Array[String]): Unit = {
    in.
      via(double_flow).
      take(10).
      runWith(print_sink).
      onComplete(_ => system.terminate)
  }
}
