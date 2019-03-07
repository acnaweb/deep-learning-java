package br.com.marketmining.dl4j.sample;

public interface Net {

	void execute();

	void evaluate();

	void buildNet();

	void buildDataSet();

	void fit();

}