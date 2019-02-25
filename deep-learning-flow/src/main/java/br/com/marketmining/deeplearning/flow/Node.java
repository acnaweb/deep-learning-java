package br.com.marketmining.deeplearning.flow;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public abstract class Node {
	protected List<Node> inputs;
	protected List<Node> outputs;
	protected INDArray value;
	protected String name;

	// input node
	public Node(String name) {
		this.name = name;
		this.value = Nd4j.zeros(1);
		this.outputs = new ArrayList<Node>();
	}

	// intermediate node
	public Node(String name, List<Node> inputs) {
		this.name = name;
		this.inputs = inputs;

		// init zero value
		this.value = Nd4j.zeros(1);

		this.outputs = new ArrayList<Node>();

		// register self for updates
		for (Node node : inputs) {
			node.outputs.add(this);
		}
	}

	public abstract void forward();

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("Node [value=");
		builder.append(value);
		builder.append(", name=");
		builder.append(name);
		builder.append("]");
		return builder.toString();
	}

}
