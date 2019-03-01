package br.com.marketmining.deeplearning.flow;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Mse extends Node {
	private final int NODE_INPUT_CALCULATED = 0;
	private final int NODE_INPUT_Y = 1;

	// y - ŷ
	private INDArray diff;

	// instantes quantity
	private long n;

	public Mse(String name, Node outputCalculated, Node outputY) {
		super(name, Arrays.asList(outputCalculated, outputY));
	}

	@Override
	public void forward() {
		INDArray outputCalculatedValue = this.inputs.get(NODE_INPUT_CALCULATED).value;
		INDArray outputYValue = this.inputs.get(NODE_INPUT_Y).value;
		this.diff = outputCalculatedValue.sub(outputYValue);
		this.n = outputCalculatedValue.slices();

		this.value = Nd4j.mean(diff.mul(diff));

	}

	@Override
	public void backward() {
		this.gradients.put(this.inputs.get(NODE_INPUT_CALCULATED), this.diff.mul(2 / this.n));
		this.gradients.put(this.inputs.get(NODE_INPUT_Y), this.diff.mul(-2 / this.n));

	}

}
