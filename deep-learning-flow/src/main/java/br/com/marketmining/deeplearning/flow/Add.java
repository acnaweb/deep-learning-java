package br.com.marketmining.deeplearning.flow;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Add extends Node {
	private final int NODE_X = 0;
	private final int NODE_Y = 1;

	public Add(Node x, Node y) {
		super("Add", Arrays.asList(x, y));
	}

	@Override
	public void forward() {
		INDArray xValue = this.inputs.get(NODE_X).value;
		INDArray yValue = this.inputs.get(NODE_Y).value;
		this.value = xValue.add(yValue);
	}

}
