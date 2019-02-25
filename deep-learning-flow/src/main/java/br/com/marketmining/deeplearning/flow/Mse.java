package br.com.marketmining.deeplearning.flow;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Mse extends Node {
	private final int NODE_INPUT_CALCULATED = 0;
	private final int NODE_INPUT_EXPECTED = 1;

	public Mse(Node outuputCalculated, Node outputExpected) {
		super("MSE", Arrays.asList(outuputCalculated, outputExpected));
	}

	@Override
	public void forward() {
		INDArray outputCalculatedValue = this.inputs.get(NODE_INPUT_CALCULATED).value;
		INDArray outputExpectedValue = this.inputs.get(NODE_INPUT_EXPECTED).value;
		INDArray diff = outputCalculatedValue.sub(outputExpectedValue);
		
		this.value = Nd4j.mean(diff.mul(diff));

	}

}
