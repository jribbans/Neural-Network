import scala.Array

object Testing {
	def main(args: Array[String]) {
		val trainingData = Array[(Double, Double)]((1,1), (2,2), (4,5), (-3,4), (9,9), (-10,5), (3,6), (5,5), (7,10))
		val trainingLabels = Array[(Double, Double)]((1,0), (1,0), (0,1), (0,1), (1,0), (0,1), (0,1), (1,0), (0,1))

		var network = new Network()

		network.addInputLayer(2, "InputLogisticLayer")
		network.addHiddenLayer(4, "HiddenLogisticLayer")
		network.addOutputLayer(2, "OutputLogisticLayer")

		for (i <- 0 until trainingData.size) {
			network.addTrainingData(Array[Double](trainingData(i)._1, trainingData(i)._2), 
									Array[Double](trainingLabels(i)._1, trainingLabels(i)._2))
		}

		network.connectNetwork()

		for (i <- 1 to 10) {
			network.train()
			val results = network.testTrainingData()
			println(results._1/(results._1 + results._2))
		}
	}
}