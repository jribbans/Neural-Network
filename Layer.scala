import scala.Array

abstract class Layer(var nodes: Array[SimpleNeuron]) {

	var backwardNodesLayer: Layer = null.asInstanceOf[Layer]
	var forwardNodesLayer: Layer = null.asInstanceOf[Layer]
	var forwardWeightsArray: Array[Array[Double]] = null.asInstanceOf[Array[Array[Double]]]
	var backwardWeightsArray: Array[Array[Double]] = null.asInstanceOf[Array[Array[Double]]]

	def connectBackwardNodes(_backwardsLayer: Layer)
	def connectForwardNodes(_forwardsLayer: Layer)
	def forwardPropogation() 
	def backwardPropogation(): Array[Array[Double]]
	
	def transformationFunction(x: Double): Double

	def weightInitializationScheme(): Double
}