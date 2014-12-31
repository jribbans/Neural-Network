import scala.Array

abstract class InputLayer(nodes: Array[SimpleNeuron]) extends Layer(nodes) {

	def connectForwardNodes(_forwardNodesLayer: Layer) {
		forwardNodesLayer = _forwardNodesLayer
		forwardWeightsArray = Array.fill[Double](forwardNodesLayer.nodes.size, 
													nodes.size)(weightInitializationScheme)
		forwardNodesLayer.connectBackwardNodes(this)
	}

	def forwardPropogation(inputs: Array[Double]) = {
		for (i <- 0 until nodes.size) {
			nodes(i).output = transformationFunction(inputs(i))
		}
	}

	def backwardPropogation(): Array[Array[Double]] = {
		var deltaWeightsArray: Array[Array[Double]] = 
						Array.fill[Double](forwardNodesLayer.nodes.size, nodes.size)(0.0)
		val forwardNodes = forwardNodesLayer.nodes
		for (i <- 0 until nodes.size) {
			var totalError: Double = 0
			for (j <- 0 until forwardNodes.size) {
					totalError += forwardNodes(j).errorTerm * forwardWeightsArray(j)(i) * 
										nodes(i).output * (1 - nodes(i).output)
					deltaWeightsArray(j)(i) = nodes(i).output * forwardNodes(j).errorTerm
				}
			nodes(i).errorTerm = totalError
		}
		return deltaWeightsArray
	}

	@deprecated("Should never be called")
	def connectBackwardNodes(_backwardsLayer: Layer) = {}
	@deprecated("Should never be called")
	def forwardPropogation() = {}
}