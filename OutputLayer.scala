import scala.Array

abstract class OutputLayer(nodes: Array[SimpleNeuron]) extends Layer(nodes){

	def connectBackwardNodes(_backwardNodesLayer: Layer) {
		backwardNodesLayer = _backwardNodesLayer
		backwardWeightsArray = backwardNodesLayer.forwardWeightsArray
	}

	def forwardPropogation() {
		for (j <- 0 until nodes.size) {
			var value: Double = 0
			for (i <- 0 until backwardNodesLayer.nodes.size) {
				value += backwardNodesLayer.nodes(i).output * backwardWeightsArray(i)(j)
			}
			nodes(j).output = transformationFunction(value)
		}
	}

	def backwardPropogation(labels: Array[Double]) = {
		for (i <- 0 until nodes.size) {
			nodes(i).errorTerm = (labels(i) - nodes(i).output) * nodes(i).output * (1 - nodes(i).output)
		}
	}

	@deprecated("Should never be called")
	def connectForwardNodes(_forwardsLayer: Layer) = {}
	@deprecated("Should never be called")
	def backwardPropogation(): Array[Array[Double]] = {null.asInstanceOf[Array[Array[Double]]]}
	@deprecated("Should never be called")
	def weightInitializationScheme(): Double 
}