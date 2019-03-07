package br.com.marketmining.dl4j.sample;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;

public abstract class AbstractNet implements Net {

	protected MultiLayerConfiguration configuration;
	protected MultiLayerNetwork model;
	protected DataSet ds;
	protected int quantityInputNeurons;
	protected int quantityLabelNeurons;
	protected int quantityHiddenNeurons;

	@Override
	public void execute() {
		this.buildDataSet();
		this.buildNet();
		this.fit();
		this.evaluate();

	}

}
