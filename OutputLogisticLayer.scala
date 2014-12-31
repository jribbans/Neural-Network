import scala.util.Random

class OutputLogisticLayer(nodes: Array[SimpleNeuron]) extends OutputLayer(nodes) {
	def transformationFunction(x: Double): Double = 1.0/ (1.0 + math.exp(-x))
	def weightInitializationScheme(): Double = Random.nextGaussian()
}