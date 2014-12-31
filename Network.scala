import scala.collection.mutable.ListBuffer
import java.lang.RuntimeException


class Network(	var inputLayer: InputLayer, 
				var hiddenLayers: ListBuffer[HiddenLayer], 
				var outputLayer: OutputLayer, 
				var trainingData: ListBuffer[Array[Double]], 
				var trainingLabels: ListBuffer[Array[Double]], 
				var testData: ListBuffer[Array[Double]], 
				var testLabels: ListBuffer[Array[Double]], 
				var alpha: Double, 
				var lambda: Double) {
	
	def this() = {
		this(null.asInstanceOf[InputLayer], new ListBuffer[HiddenLayer](), 
			null.asInstanceOf[OutputLayer], new ListBuffer[Array[Double]](), 
			new ListBuffer[Array[Double]](), new ListBuffer[Array[Double]](), 
			new ListBuffer[Array[Double]](), 1/3, 1.3)
	}

	def addHiddenLayer(n: Int, t: String) = {
		var neurons = Array.fill[SimpleNeuron](n)(new SimpleNeuron())
		t match {
			case "HiddenLogisticLayer" => hiddenLayers.append(new HiddenLogisticLayer(neurons))
			case _ => throw new RuntimeException("Not a valid type of HiddenLayer")
		}
	}

	def addInputLayer(n: Int, t: String) = {
		var neurons = Array.fill[SimpleNeuron](n)(new SimpleNeuron())
		t match {
			case "InputLogisticLayer" => inputLayer = new InputLogisticLayer(neurons)
			case _ => throw new RuntimeException("Not a valid type of HiddenLayer")
		}
	}

	def addOutputLayer(n: Int, t: String) = {
		var neurons = Array.fill[SimpleNeuron](n)(new SimpleNeuron())
		t match {
			case "OutputLogisticLayer" => outputLayer = new OutputLogisticLayer(neurons)
			case _ => throw new RuntimeException("Not a valid type of HiddenLayer")
		}
	}

	def connectNetwork() = {
		inputLayer.connectForwardNodes(hiddenLayers(0))
		for (i <- 0 until hiddenLayers.size - 1) {
			hiddenLayers(i).connectForwardNodes(hiddenLayers(i+1))
		}
		hiddenLayers(hiddenLayers.size - 1).connectForwardNodes(outputLayer)
	}

	def addTrainingData(_trainingData: Array[Double], _trainingLabels: Array[Double]) = {
		trainingData.append(_trainingData)
		trainingLabels.append(_trainingLabels)
	}

	def addTestData(_testData: Array[Double], _testLabels: Array[Double]) = {
		testData.append(_testData)
		testLabels.append(_testLabels)
	}

	def train() = {
		var delta: ListBuffer[Array[Array[Double]]] = new ListBuffer[Array[Array[Double]]]()
		
		inputLayer.forwardPropogation(trainingData(0))
		for (i <- 0 until hiddenLayers.size) {
			hiddenLayers(i).forwardPropogation()
		}
		outputLayer.forwardPropogation()
		outputLayer.backwardPropogation(trainingLabels(0))
		for (i <- hiddenLayers.size - 1 to 0) {
			delta.prepend(hiddenLayers(i).backwardPropogation())
		}
		delta.prepend(inputLayer.backwardPropogation())

		for (d <- 1 to trainingData.size) {
			inputLayer.forwardPropogation(trainingData(d))
			for (i <- 0 until hiddenLayers.size) {
				hiddenLayers(i).forwardPropogation()
			}
			outputLayer.forwardPropogation()
			outputLayer.backwardPropogation(trainingLabels(d))
			for (i <- hiddenLayers.size - 1 to 0) {
				addMatrices(delta(i+1), hiddenLayers(i).backwardPropogation())
			}
			addMatrices(delta(0), inputLayer.backwardPropogation())
		}

		inputLayer.forwardWeightsArray = addSpecialMatrices(inputLayer.forwardWeightsArray, delta(0), trainingData.size)
		for (i <- 0 until hiddenLayers.size) {
			hiddenLayers(i).forwardWeightsArray = addSpecialMatrices(hiddenLayers(i).forwardWeightsArray, delta(i+1), trainingData.size)
		} 
	}

	def addMatrices(a: Array[Array[Double]], b: Array[Array[Double]]): Array[Array[Double]] = {
		for (i <- 0 until a.size) {
			for (j <- 0 until a(0).size) {
				a(i)(j) = a(i)(j) + b(i)(j)
			}
		}
		return a
	}

	def addSpecialMatrices(a: Array[Array[Double]], b: Array[Array[Double]], m: Int): Array[Array[Double]] = {
		for (i <- 0 until a.size) {
			for (j <- 0 until a(0).size) {
				a(i)(j) = a(i)(j) - alpha * ((b(i)(j)/m) + lambda * a(i)(j))
			}
		}
		return a
	}

	def evaluateCorrect(output: Array[SimpleNeuron], trainingLabels: Array[Double]): Int = {
		for (i <- 0 until output.size) {
			val bool = output(i).output + trainingLabels(i)
			if (((bool > 1.5) || (bool < .5))) {
				return 1
			}
		}
		return 0
	}

	def testTrainingData(): (Int, Int) = {
		var incorrect: Int = 0
		for (d <- 1 to trainingData.size) {
			inputLayer.forwardPropogation(trainingData(d))
			for (i <- 0 until hiddenLayers.size) {
				hiddenLayers(i).forwardPropogation()
			}
			evaluateCorrect(outputLayer.nodes, trainingLabels(d))
		}
		return (trainingData.size - incorrect, incorrect)
	}
}