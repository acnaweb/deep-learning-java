package br.com.marketmining.deeplearning.perceptron;

public class RunPerceptron {
	
	/**********************************************************************/
	public static void main(String[] args) {
		new RunPerceptron().runFixed();
	}
	
	/**********************************************************************/
	public void runFixed() {
		// weights
		double[] w;

		// quantity inputs
		int n;

		/**************************************/
		// neuronio AND
		// bias, w1, w2
		w = new double[] { -3, 2, 2 };
		n = 2;
		Perceptron and = new Perceptron(w, n);

		/**************************************/
		// neuronio OR
		// bias, w1, w2
		w = new double[] { -3, 4, 4 };
		n = 2;
		Perceptron or = new Perceptron(w, n);

		/**************************************/
		// neuronio NOT
		// bias, w1
		w = new double[] { 1, -2 };
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
			int resultNot = not.calculate(new int[] { x[0] });

			// XOR
			int resultNotAnd = not.calculate(new int[] { resultAnd });
			int resultXor = and.calculate(new int[] { resultOr, resultNotAnd });

			System.out.println(
					x[0] + "\t" + x[1] + "\t" + resultAnd + "\t" + resultOr + "\t" + resultXor + "\t" + resultNot);

		}
	}

	
}
