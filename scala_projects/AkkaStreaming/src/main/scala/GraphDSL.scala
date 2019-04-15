import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import akka.stream.scaladsl._
import akka.stream.{ClosedShape, FlowShape}
import GraphDSL.Implicits._
import scala.concurrent.duration._
import scala.concurrent.Future

object GraphDsl extends App {
  implicit val system = ActorSystem("Streaming")
  implicit val executor = system.dispatcher
  implicit val materializer = ActorMaterializer()

  val graph = RunnableGraph.fromGraph(GraphDSL.create() { implicit builder =>

    Streaming.in ~> Streaming.double_flow ~> Flow[Int].take(10) ~> Streaming.print_sink

    ClosedShape
  })

  val partial_graph = Flow.fromGraph(GraphDSL.create() { implicit builder =>
    val broadcast = builder.add(Broadcast[Int](2))
    val zip = builder.add(Zip[Int, Int]())

    broadcast.out(0) ~> Flow[Int].filter(_ % 2 == 0) ~> Flow[Int].map(_ / 2) ~> zip.in0
    broadcast.out(1) ~> Flow[Int].filter(_ % 2 != 0) ~> Flow[Int].map(_ * 2) ~> zip.in1

    FlowShape(broadcast.in, zip.out)
  })

  val merge_graph = RunnableGraph.fromGraph(GraphDSL.create() { implicit builder =>
    val merge = builder.add(Merge[Int](2))
  
    Streaming.in ~> Streaming.double_flow ~> merge ~> Streaming.print_sink
                             Streaming.in ~> merge

    ClosedShape
  })

}
