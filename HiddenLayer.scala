import scala.Array

abstract class HiddenLayer(nodes: Array[SimpleNeuron]) extends Layer(nodes) {

	def connectForwardNodes(_forwardNodesLayer: Layer) {
		forwardNodesLayer = _forwardNodesLayer
		forwardWeightsArray = Array.fill[Double](forwardNodesLayer.nodes.size, 
													nodes.size)(weightInitializationScheme)
		forwardNodesLayer.connectBackwardNodes(this)
	}

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
}