import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import akka.stream.scaladsl._
import akka.stream.{ClosedShape, FlowShape}
import GraphDSL.Implicits._
import scala.concurrent.duration._
import scala.concurrent.Future

object ZippingStreams {
  implicit val system = ActorSystem("Streaming")
  implicit val executor = system.dispatcher
  implicit val materializer = ActorMaterializer()

  val naturalNumbers = Source.unfold(1)(a => Some(a + 1, a))

  val in = Source.tick(1.second, 10.seconds, 1)

  val zipStream = RunnableGraph.fromGraph(GraphDSL.create() { implicit builder =>
    val zip = builder.add(Zip[Int, Int])

    naturalNumbers ~> zip.in0
                in ~> zip.in1

    zip.out ~> Streaming.print_sink

    ClosedShape
  })

  def main(args: Array[String]): Unit = {
    zipStream.run()
  }
}
