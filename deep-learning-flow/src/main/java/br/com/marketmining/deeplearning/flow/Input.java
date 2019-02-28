package br.com.marketmining.deeplearning.flow;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Input extends Node {

	public Input(String name) {
		super(name);
	}

	@Override
	public void forward() {
	}

	@Override
	public void backward() {
		this.gradients.put(this, Nd4j.zeros(this.value.shape()));

		// calculate for ouputs
		for (Node output : this.outputs) {
			// get gradiente value calculated on outputs for this node
			INDArray gradientOutputValue = output.gradients.get(this);
			
			this.gradients.get(this).addi(gradientOutputValue);
		}
	}

}
