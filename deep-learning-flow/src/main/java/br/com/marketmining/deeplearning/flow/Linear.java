package br.com.marketmining.deeplearning.flow;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Linear extends Node {
	private final int NODE_INPUTS = 0;
	private final int NODE_WEIGHTS = 1;
	private final int NODE_BIAS = 2;

	public Linear(Node inputs, Node weights, Node bias) {
		super("Linear", Arrays.asList(inputs, weights, bias));
	}

	@Override
	public void forward() {
		INDArray inputValues = this.inputs.get(NODE_INPUTS).value;
		INDArray weightValues = this.inputs.get(NODE_WEIGHTS).value;
		INDArray biasValue = this.inputs.get(NODE_BIAS).value;
		
		// linear combination
		this.value = inputValues.mmul(weightValues).addRowVector(biasValue);

	}

}
