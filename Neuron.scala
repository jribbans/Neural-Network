import scala.collection.mutable.ListBuffer

abstract class Neuron(var backwardConnections: ListBuffer[Connection], 
	var forwardConnections: ListBuffer[Connection], var value: Double) {

	def this() {
		this(new ListBuffer[Connection](), new ListBuffer[Connection](), 0)
	}

	def addBackConnection(x: Connection) {
		backConnections.append(x)
	}

	def addForwardConnection(x: Connection) {
		forwardConnections.append(x)
	}

	def forwardPropogation() {
		var total: Double = 0
		for (connection <- backwardConnections) {
			total += connection.edgeWeight * connection.value
		}
		this.value = transformationFunction(total)
		val value = transformationFunction(total) 
		for (connection <- forwardConnections) {
			connection.value = value
		}
	}

	def backwardPropogation(): Double {
		var totalError: Double = 0
		for (connection <- forwardConnections) {
			totalError += connection.error * connection.value * this.value * (1 - this.value)
		}
		partialConnection = totalError * connection.value
		for (connection <- backwardConnections) {
			connection.error = totalError
		}
		return partialConnection
	}

	def transformationFunction(x: Double): Double
}