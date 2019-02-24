package br.com.marketmining.deeplearning.perceptron;

import java.util.Random;

public class Perceptron {
	private static final int INDEX_BIAS = 0;

	// weights
	private double[] w;

	// quantity inputs
	private int n;

	/**********************************************************************/
	public Perceptron(int n) {
		this.n = n;
		this.initWeights();
	}

	/**********************************************************************/
	public Perceptron(double[] w, int n) {
		this.w = w;
		this.n = n;
	}

	/**********************************************************************/
	private void initWeights() {
		Random r = new Random(42);
		this.w = new double[n + 1];

		for (int i = 0; i <= n; i++) {
			this.w[i] = r.nextDouble();
		}
	}

	/**********************************************************************/
	public int calculate(int x[]) {
		if (x.length != n)
			throw new IllegalArgumentException("x.length != n");

		// h - linear combination
		double h = w[INDEX_BIAS];

		for (int i = 0; i < n; i++) {
			h += x[i] * w[i + 1];
		}

		return this.stepFunction(h);
	}

	/**********************************************************************/
	private int stepFunction(double h) {
		int result;
		if (h < 0)
			result = 0;
		else
			result = 1;
		return result;
	}



	
}
