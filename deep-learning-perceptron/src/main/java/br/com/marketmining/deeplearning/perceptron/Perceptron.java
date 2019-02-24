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

	/**
	 * @param dataset
	 * @param y
	 * @param learningTax
	 * @param epochs
	 */
	public void training(double dataset[][], double y[], float learningTax, int epochs) {

		int datasetSize = dataset.length;

		for (int currentEpoch = 1; currentEpoch <= epochs; currentEpoch++) {

			for (int i = 0; i < datasetSize; i++) {
				double x[] = dataset[i];

				double yCalculated = calculate(x);
				double yExpected = y[i];
				double error = this.calculateError(yCalculated, yExpected);

				// System.out.println("yCalculated=" + yCalculated + "\tyExpected=" + yExpected
				// + "\terror=" + error);
				this.updateWeights(yExpected, error, learningTax);
			}
		}
	}

	/**
	 * @param yExpected
	 * @param error
	 * @param learningTax
	 */
	private void updateWeights(double yExpected, double error, float learningTax) {
		// peso - (entrada * erro * taxa de aprendizagem)

		for (int i = 0; i < w.length; i++) {
			if (i == 0) {// bias
				w[i] = w[i] - error * learningTax;
			} else {
				w[i] = w[i] - yExpected * error * learningTax;
			}
		}

	}

	/**
	 * @param yCalculated
	 * @param yExpected
	 * @return
	 */
	private double calculateError(double yCalculated, double yExpected) {
		return yCalculated - yExpected;
	}

	/**
	 * @param x
	 * @return
	 */
	public double calculate(double x[]) {
		if (x.length != n)
			throw new IllegalArgumentException("x.length != n");

		// h - linear combination
		double h = w[INDEX_BIAS];

		for (int i = 0; i < n; i++) {
			h += x[i] * w[i + 1];
		}

		return this.stepFunction(h);
	}

	/**
	 * 
	 */
	private void initWeights() {
		Random r = new Random();
		this.w = new double[n + 1];

		for (int i = 0; i <= n; i++) {
			this.w[i] = r.nextDouble();
		}
	}

	/**
	 * 
	 */
	public void printWeights() {
		String msg = "";

		for (int i = 0; i <= n; i++) {
			msg += "w[" + i + "]=" + this.w[i] + "\t";
		}

		System.out.println(msg);
	}

	/**
	 * @param h
	 * @return
	 */
	private int stepFunction(double h) {
		int result;
		if (h < 0)
			result = 0;
		else
			result = 1;
		return result;
	}

}
