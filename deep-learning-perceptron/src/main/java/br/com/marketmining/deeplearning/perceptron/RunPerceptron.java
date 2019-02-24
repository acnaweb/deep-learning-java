package br.com.marketmining.deeplearning.perceptron;

public class RunPerceptron {

	/**********************************************************************/
	public static void main(String[] args) {
		// new RunPerceptron().runFixed();
		new RunPerceptron().runMain();

	}

	public void runMain() {
		int n = 2;
		float learningTax = (float) 0.1;
		int epochs = 1000;
		double[][] dataset = new double[4][2];
		dataset[0] = new double[] { 1, 1 };
		dataset[1] = new double[] { 1, 0 };
		dataset[2] = new double[] { 0, 1 };
		dataset[3] = new double[] { 0, 0 };

		// and
		System.out.println("Training And");
		Perceptron and = new Perceptron(n);
		double[] yAnd = { 1, 0, 0, 0 };
		and.training(dataset, yAnd, learningTax, epochs);
		and.printWeights();

		// or
		System.out.println("Training Or");
		Perceptron or = new Perceptron(n);
		double[] yOr = { 1, 1, 1, 0 };
		or.training(dataset, yOr, learningTax, epochs);
		or.printWeights();

		// not
		System.out.println("Training Not");
		n = 1;
		double[][] datasetOr = new double[2][1];
		datasetOr[0] = new double[] { 1 };
		datasetOr[1] = new double[] { 0 };

		Perceptron not = new Perceptron(n);
		double[] yNot = { 0, 1 };
		not.training(datasetOr, yNot, learningTax, epochs);
		not.printWeights();

		/**************************************/
		// calculating
		System.out.println("x[0]\tx[1]\tAND\tOR\tXOR\tNOTx[0]");
		for (int i = 0; i < dataset.length; i++) {
			// input
			double[] x = dataset[i];

			double resultAnd = and.calculate(x);
			double resultOr = or.calculate(x);
			double resultNot = not.calculate(new double[] { x[0] });

			// XOR
			double resultNotAnd = not.calculate(new double[] { resultAnd });
			double resultXor = and.calculate(new double[] { resultOr, resultNotAnd });

			System.out.println(
					x[0] + "\t" + x[1] + "\t" + resultAnd + "\t" + resultOr + "\t" + resultXor + "\t" + resultNot);

		}

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
		// dataset
		double[][] dataset = new double[4][2];
		dataset[0] = new double[] { 1, 1 };
		dataset[1] = new double[] { 1, 0 };
		dataset[2] = new double[] { 0, 1 };
		dataset[3] = new double[] { 0, 0 };

		/**************************************/
		// calculating
		System.out.println("x[0]\tx[1]\tAND\tOR\tXOR\tNOTx[0]");
		for (int i = 0; i < dataset.length; i++) {
			// input
			double[] x = dataset[i];

			double resultAnd = and.calculate(x);
			double resultOr = or.calculate(x);
			double resultNot = not.calculate(new double[] { x[0] });

			// XOR
			double resultNotAnd = not.calculate(new double[] { resultAnd });
			double resultXor = and.calculate(new double[] { resultOr, resultNotAnd });

			System.out.println(
					x[0] + "\t" + x[1] + "\t" + resultAnd + "\t" + resultOr + "\t" + resultXor + "\t" + resultNot);

		}
	}

}
