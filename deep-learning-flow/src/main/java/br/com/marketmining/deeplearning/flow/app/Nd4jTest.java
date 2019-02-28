package br.com.marketmining.deeplearning.flow.app;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Nd4jTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		INDArray x = Nd4j.ones(4, 1);
		System.out.println(x.shape().length);

		INDArray y = Nd4j.ones(4);

		System.out.println(x);
		System.out.println(y);

		INDArray mmul = x.mmul(y);
		System.out.println(mmul);

		INDArray z = Nd4j.ones(2, 2);

		INDArray w = Nd4j.ones(2, 2);

		INDArray mul = z.add(w);
		INDArray mean = Nd4j.mean(mul);

		System.out.println("mul");

		System.out.println(mul);

		System.out.println("mean");

		System.out.println(mean);

		double[][] values = { { 1, 3, 1, 4 }, { 2, 3, 1, 4 }, { 3, 5, 5, 5 }, { 4, 5, 5, 5 }, { 5, 5, 5, 5 } };
		INDArray ind = Nd4j.create(values);
		System.out.println(ind.isMatrix());
		System.out.println(ind.isVector());
		System.out.println(ind.isScalar());
		System.out.println(ind.isRowVector());
		System.out.println(ind.slices());
		System.out.println(ind.shape()[1]);

		System.out.println(values[0][1]);

		System.out.println("aritmetica");

		INDArray val1 = Nd4j
				.create(new double[][] { { 2.0, 2.0, 6.0 }, { 2.0, 4.0, 6.0 }, { 2.0, 2.0, 6.0 }, { 2.0, 4.0, 6.0 } });
		INDArray val2 = Nd4j
				.create(new double[][] { { 1.0, 1.0, 5.0 }, { 0.0, 1.0, 1.5 }, { 1.0, 1.0, 5.0 }, { 0.0, 1.0, 1.5 } });

		INDArray result = val1.sub(val2);
		System.out.println(result);
		System.out.println(result.slices());
		
		System.out.println((2.0 / result.slices()));
		System.out.println((-2.0 / result.slices()));
		
		System.out.println(result.mul((2.0 / result.slices())));
		
		System.out.println(result.mul((-2.0 / result.slices())));
		
		INDArray novo = Nd4j.ones(result.shape());
		INDArray novo1 = Nd4j.ones(result.shape());
		result = novo.addi(novo1);
		result = novo.addi(novo1);

		System.out.println("***************************");
		System.out.println(novo);
		System.out.println(novo1);
		System.out.println(result);
		
		System.out.println("***************************");
		System.out.println(val2);
		System.out.println(val2.transpose());
		System.out.println(Nd4j.sum(val2));

		


	}
}
