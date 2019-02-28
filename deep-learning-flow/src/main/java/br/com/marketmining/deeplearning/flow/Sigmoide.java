package br.com.marketmining.deeplearning.flow;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

	@Override
	public void backward() {

		// calculate for ouputs
		for (Node output : this.outputs) {
			// get gradiente value calculated on outputs for this node
			INDArray gradientOutputValue = output.gradients.get(this);

			// value(s) for input(s)
			// linear combination
			INDArray hValue = this.inputs.get(NODE_H).value;

			// calculate sigmoid
			INDArray sigmoidValue = this.sigmoide(hValue);

			// deriving sigmoid
			INDArray vetorOnes = Nd4j.ones(hValue.slices());
			INDArray derivedSigmoid = sigmoidValue.mul(vetorOnes.sub(sigmoidValue));

			// gradients x derived sigmoid
			INDArray g = gradientOutputValue.mul(derivedSigmoid);

			this.gradients.get(this.inputs.get(NODE_H)).addi(g);
		}
	}

}
