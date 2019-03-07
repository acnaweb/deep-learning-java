package br.com.marketmining.dl4j.sample;

public class Main {
	public static void main(String[] args) {
		Net xor = new Xor();
		// xor.execute();

		Net mnist = new Mnist();
		mnist.execute();
	}
}
