class LogisticNeuron extends Neuron {
	def transformationFunction(x: Double): Double = 1.0/ (1.0 + math.exp(-x))
}
