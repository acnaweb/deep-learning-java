package br.com.marketmining.deeplearning.flow;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Linear extends Node {
	private final int NODE_INPUTS = 0;
	private final int NODE_WEIGHTS = 1;
	private final int NODE_BIAS = 2;

	public Linear(String name, Node inputs, Node weights, Node bias) {
		super(name, Arrays.asList(inputs, weights, bias));
	}

	@Override
	public void forward() {
		INDArray inputValues = this.inputs.get(NODE_INPUTS).value;
		INDArray weightValues = this.inputs.get(NODE_WEIGHTS).value;
		INDArray biasValue = this.inputs.get(NODE_BIAS).value;

		// linear combination
		this.value = inputValues.mmul(weightValues).addRowVector(biasValue);

	}

	@Override
	public void backward() {
		/// initializing gradientes for inputs
		for (Node input : inputs) {
			this.gradients.put(input, Nd4j.zeros(input.value.shape()));
		}

		// calculate for ouputs
		for (Node output : this.outputs) {
			// get gradiente value calculated on outputs for this node
			INDArray gradientOutputValue = output.gradients.get(this);

			// value(s) for input(s)
			INDArray inputValues = this.inputs.get(NODE_INPUTS).value;
			INDArray weightValues = this.inputs.get(NODE_WEIGHTS).value;
			INDArray biasValue = this.inputs.get(NODE_BIAS).value;

			this.gradients.get(this.inputs.get(NODE_INPUTS)).addi(gradientOutputValue.mmul(weightValues.transpose()));
			this.gradients.get(this.inputs.get(NODE_WEIGHTS)).addi(inputValues.transpose().mmul(gradientOutputValue));
			this.gradients.get(this.inputs.get(NODE_BIAS)).addi(Nd4j.sum(biasValue));

		}
	}

}
