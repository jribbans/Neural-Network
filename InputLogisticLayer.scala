import scala.util.Random

class InputLogisticLayer(nodes: Array[SimpleNeuron]) extends InputLayer(nodes) {
	def transformationFunction(x: Double): Double = 1.0/ (1.0 + math.exp(-x))
	def weightInitializationScheme(): Double = Random.nextGaussian()
}