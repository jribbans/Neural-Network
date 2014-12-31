import scala.util.Random

class HiddenLogisticLayer(nodes: Array[SimpleNeuron]) extends HiddenLayer(nodes) {
	def transformationFunction(x: Double): Double = 1.0/ (1.0 + math.exp(-x))
	def weightInitializationScheme(): Double = Random.nextGaussian()
}