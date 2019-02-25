package br.com.marketmining.deeplearning.flow;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NeuralNetwork {

	public static void main(String[] args) {
		new NeuralNetwork().somaDois();
	}

	public void somaDois() {

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
		ArrayList<Node> sortedNodes = sortNodes(feed);

		System.out.println(sortedNodes);
		
		INDArray result = callForward(f, sortedNodes);

		System.out.println(result);

	}

	// Método que faz a passada para frente
	public static INDArray callForward(Node output, ArrayList<Node> sortedNodes) {
		// Para cada nó dentre os nós já ordenados
		for (Node n : sortedNodes) {
			// Executa o método foward do nó
			n.forward();
		}
		// Retorna o valor do último nó da rede
		return output.value;
	}

	// Método que ordena os nós usando o algoritmo de Kahn
	// https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm
	public static ArrayList<Node> sortNodes(Map<Node, INDArray> feed) {
		Set<Node> nos_Input = feed.keySet();
		ArrayList<Node> nos = new ArrayList<Node>(nos_Input);
		Map<Node, Map<String, HashSet<Node>>> G = new HashMap<Node, Map<String, HashSet<Node>>>();

		while (nos.size() > 0) {
			Node n = (Node) nos.get(0);
			nos.remove(0);
			if (!G.containsKey(n)) {
				Map<String, HashSet<Node>> in_out = new HashMap<String, HashSet<Node>>();
				HashSet<Node> in = new HashSet<Node>();
				HashSet<Node> out = new HashSet<Node>();
				in_out.put("in", in);
				in_out.put("out", out);
				G.put(n, in_out);
			}

			for (int j = 0; j < n.outputs.size(); j++) {
				Node m = n.outputs.get(j);
				if (!G.containsKey(m)) {
					HashMap<String, HashSet<Node>> in_out = new HashMap<String, HashSet<Node>>();
					HashSet<Node> in = new HashSet<Node>();
					HashSet<Node> out = new HashSet<Node>();
					in_out.put("in", in);
					in_out.put("out", out);
					G.put(m, in_out);
				}
				(G.get(n).get("out")).add(m);
				(G.get(m).get("in")).add(n);
				nos.add(m);
			}
		}
		HashSet<Node> S = new HashSet<Node>(nos_Input);
		ArrayList<Node> L = new ArrayList<Node>();
		while (S.size() > 0) {
			Node n = S.iterator().next();
			if (n instanceof Input) {
				n.value = feed.get(n);
			}
			L.add(n);
			for (int j = 0; j < n.outputs.size(); j++) {
				Node m = n.outputs.get(j);
				G.get(n).get("out").remove(m);
				(G.get(m).get("in")).remove(n);
				if (G.get(m).get("in").size() == 0) {
					S.add(m);
				}
			}
			S.remove(n);
		}
		return L;
	}

}
