package br.com.marketmining.dsaflow;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author ac
 *
 */
/**
 * @author ac
 *
 */
public abstract class Node {
	protected List<Node> inputs;
	protected List<Node> outputs;
	public INDArray value;

	protected Map<Node, INDArray> gradients;
	protected String name;

	// input node
	public Node(String name) {
		this.name = name;
		this.value = Nd4j.zeros(1);
		this.outputs = new ArrayList<Node>();
		this.gradients = new HashMap<Node, INDArray>();
	}

	// intermediate node
	public Node(String name, List<Node> inputs) {
		this.name = name;
		this.inputs = inputs;

		// init zero value
		this.value = Nd4j.zeros(1);

		// init
		this.outputs = new ArrayList<Node>();
		this.gradients = new HashMap<Node, INDArray>();

		this.connectSelfOnInputs();
	}

	private void connectSelfOnInputs() {
		// connect self for updates
		for (Node node : inputs) {
			node.outputs.add(this);
		}
	}

	public abstract void forward();

	public abstract void backward();

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
