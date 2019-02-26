package br.com.marketmining.deeplearning.flow.app;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import br.com.marketmining.deeplearning.flow.Add;
import br.com.marketmining.deeplearning.flow.Input;
import br.com.marketmining.deeplearning.flow.NeuralNetwork;
import br.com.marketmining.deeplearning.flow.Node;

public class NetSomaDois {

	public void execute() {

		// input
		Input xInput = new Input("x");
		Input yInput = new Input("y");

		// intermediate (operations)
		Add f = new Add(xInput, yInput);

		// output

		// values
		double[] x = { 10 };
		double[] y = { 40 };

		INDArray xValue = Nd4j.create(x);
		INDArray yValue = Nd4j.create(y);

		// values for inputs
		Map<Node, INDArray> feed = new HashMap<Node, INDArray>();
		feed.put(xInput, xValue);
		feed.put(yInput, yValue);

		// sort nodes
		ArrayList<Node> graph = NeuralNetwork.sortNodes(feed);

		System.out.println(graph);

		INDArray result = NeuralNetwork.callForward(f, graph);

		System.out.println(result);

	}

}
