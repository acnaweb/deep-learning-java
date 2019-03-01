package br.com.marketmining.deeplearning.forward;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import br.com.marketmining.deeplearning.flow.Input;
import br.com.marketmining.deeplearning.flow.Linear;
import br.com.marketmining.deeplearning.flow.AnnUtil;
import br.com.marketmining.deeplearning.flow.Node;

public class NetLinear {

	public void execute() {
		Input inputs = new Input("x");
		Input weights = new Input("w");
		Input bias = new Input("bias");

		Linear linear = new Linear("linear", inputs, weights, bias);

		// values
		double[][] inputsValues = { { -1, -2 }, { -1, -2 } };
		double[][] weightsValues = { { 2, -3 }, { 2, -3 } };
		double[] biasValue = { -3, -5 };
		INDArray x = Nd4j.create(inputsValues);
		INDArray w = Nd4j.create(weightsValues);
		INDArray b = Nd4j.create(biasValue);

		// binding values to inputs
		Map<Node, INDArray> feed = new HashMap<Node, INDArray>();
		feed.put(inputs, x);
		feed.put(weights, w);
		feed.put(bias, b);

		ArrayList<Node> graph = AnnUtil.sortNodes(feed);

		INDArray result = AnnUtil.callForward(linear, graph);

		System.out.println(result);

	}

}
