package br.com.marketmining.deeplearning.flow;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Sigmoide extends Node {
	private final int NODE_H = 0;

	public Sigmoide(Node h) {
		super("Sigmoide", Arrays.asList(h));
	}

	private INDArray sigmoide(INDArray value) {
		return sigmoid(value);
	}

	@Override
	public void forward() {
		INDArray hValue = this.inputs.get(NODE_H).value;

		this.value = this.sigmoide(hValue);
	}

}
