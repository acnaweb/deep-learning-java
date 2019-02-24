package br.com.marketmining.deeplearning.perceptron;

public class Perceptron {
	private static final int BIAS = 1;
	private double[] w;
	private int n;

	/**********************************************************************/
	public Perceptron(double[] w, int n) {
		this.w = w;
		this.n = n;
	}

	/**********************************************************************/
	public int calculate(int x[]) {
		if (x.length != n)
			throw new IllegalArgumentException("x.length != n");

		double h = w[n] * BIAS;
		for (int i = 0; i < n; i++) {
			h += x[i] * w[i];
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

	/**********************************************************************/
	public static void main(String[] args) {

		// weights
		double[] w;

		// quantity features / net size
		int n;

		/**************************************/
		// neuronio AND
		// w1, w2, bias
		w = new double[] { 2, 2, -3 };
		n = 2;
		Perceptron and = new Perceptron(w, n);

		/**************************************/
		// neuronio OR
		// w1, w2, bias
		w = new double[] { 4, 4, -3 };
		n = 2;
		Perceptron or = new Perceptron(w, n);

		/**************************************/
		// neuronio NOT
		// w1, bias
		w = new double[] { -2, 1 };
		n = 1;
		Perceptron not = new Perceptron(w, n);

		/**************************************/
		// instances
		int[][] instances = new int[4][2];
		instances[0] = new int[] { 1, 1 };
		instances[1] = new int[] { 1, 0 };
		instances[2] = new int[] { 0, 1 };
		instances[3] = new int[] { 0, 0 };

		/**************************************/
		// calculating
		System.out.println("x[0]\tx[1]\tAND\tOR\tXOR\tNOTx[0]");
		for (int i = 0; i < instances.length; i++) {
			// instancia
			int[] x = instances[i];

			int resultAnd = and.calculate(x);
			int resultOr = or.calculate(x);
			int resultNot = not.calculate(new int[] {x[0]});

			// XOR
			int resultNotAnd = not.calculate(new int[] { resultAnd });
			int resultXor = and.calculate(new int[] { resultOr, resultNotAnd });

			System.out.println(x[0] + "\t" + x[1] + "\t" + resultAnd + "\t" + resultOr + "\t" + resultXor + "\t" + resultNot);

		}

	}
}
