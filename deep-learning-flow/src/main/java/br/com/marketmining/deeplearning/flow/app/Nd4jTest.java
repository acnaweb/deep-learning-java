package br.com.marketmining.deeplearning.flow.app;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Nd4jTest {

	public static void main(String[] args) {
		INDArray x = Nd4j.ones(4,1);
		
		INDArray y = Nd4j.ones(4);
		
		System.out.println(x);
		System.out.println(y);
		
		INDArray mmul = x.mmul(y);
		System.out.println(mmul);
		
		INDArray z = Nd4j.ones(2,2);
		
		INDArray w = Nd4j.ones(2,2);
		
		INDArray mul = z.add(w);
		INDArray mean = Nd4j.mean(mul);
		
		System.out.println("mul");

		System.out.println(mul);
		
		System.out.println("mean");

		System.out.println(mean);

	}
}
