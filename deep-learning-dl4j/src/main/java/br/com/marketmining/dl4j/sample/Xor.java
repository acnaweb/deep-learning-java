package br.com.marketmining.dl4j.sample;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Xor extends AbstractNet {

	/*
	 * (non-Javadoc)
	 * 
	 * @see br.com.marketmining.dl4j.sample.Net#evaluate()
	 */
	@Override
	public void evaluate() {
		INDArray output = this.model.output(ds.getFeatureMatrix());
		System.out.println(output);

		Evaluation eval = new Evaluation(quantityLabelNeurons);
		eval.eval(ds.getLabels(), output);
		System.out.println(eval.stats());
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see br.com.marketmining.dl4j.sample.Net#buildNet()
	 */
	@Override
	public void buildNet() {

		// criando a rede
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();

		// hiperparametros
		builder.iterations(10000);
		builder.learningRate(0.1);
		builder.seed(1234);
		builder.useDropConnect(false); // true - redes maiores
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.biasInit(0);
		builder.miniBatch(false); // true - redes maiores

		/*******************************************************************************/
		// layer input
		// layer hidder
		DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
		hiddenLayerBuilder.nIn(quantityInputNeurons);
		hiddenLayerBuilder.nOut(quantityHiddenNeurons);
		hiddenLayerBuilder.activation(Activation.SIGMOID);
		hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		hiddenLayerBuilder.dist(new UniformDistribution(0, 1));

		// layer ouput
		OutputLayer.Builder outputLayerBuilder = new OutputLayer.Builder(
				LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
		outputLayerBuilder.nIn(quantityHiddenNeurons);
		outputLayerBuilder.nOut(quantityLabelNeurons);
		// softmax -> normaliza para 1 = probabilidade de cada neuronio na output layer
		outputLayerBuilder.activation(Activation.SOFTMAX);
		outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		outputLayerBuilder.dist(new UniformDistribution(0, 1));

		/*******************************************************************************/
		// atribuindo as camadas
		ListBuilder listBuilder = builder.list();
		listBuilder.pretrain(false);
		listBuilder.backprop(true);

		listBuilder.layer(0, hiddenLayerBuilder.build());
		listBuilder.layer(1, outputLayerBuilder.build());

		this.configuration = listBuilder.build();

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see br.com.marketmining.dl4j.sample.Net#buildDataSet()
	 */
	@Override
	public void buildDataSet() {
		this.quantityInputNeurons = 2;
		this.quantityHiddenNeurons = 4;
		this.quantityLabelNeurons = 2;

		final int QUANTITY_INSTANCES = 4;

		// input layer -> 2 neuronios, entradas, atributos, features, características
		// input layer -> 4 instancias, exemplos, observacoes
		// input layer 4 x 2
		// input layer coluna 0 = valor do neuronio 1
		// input layer coluna 1 = valor do neuronio 2
		final int COL_NEURON_INPUT_1 = 0;
		final int COL_NEURON_INPUT_2 = 1;

		INDArray inputs = Nd4j.zeros(QUANTITY_INSTANCES, quantityInputNeurons);

		// output (labels) layer -> 2 n = 2 valores -
		// saida 1 (probabilidade de ser F)
		// saída 2 (probabilidade de ser V)
		final int CLASSE_F = 0;
		final int CLASSE_V = 1;
		final int LABEL_F = 1;
		final int LABEL_V = 0;

		INDArray labels = Nd4j.zeros(QUANTITY_INSTANCES, quantityLabelNeurons);

		// primeira instancia = input + label
		int linha = 0;
		inputs.putScalar(new int[] { linha, COL_NEURON_INPUT_1 }, 0);
		inputs.putScalar(new int[] { linha, COL_NEURON_INPUT_2 }, 0);

		labels.putScalar(new int[] { linha, CLASSE_F }, LABEL_F);
		labels.putScalar(new int[] { linha, CLASSE_V }, LABEL_V);

		// segunda instancia = input + label
		linha = 1;
		inputs.putScalar(new int[] { linha, COL_NEURON_INPUT_1 }, 1);
		inputs.putScalar(new int[] { linha, COL_NEURON_INPUT_2 }, 0);

		labels.putScalar(new int[] { linha, CLASSE_F }, LABEL_V);
		labels.putScalar(new int[] { linha, CLASSE_V }, LABEL_F);

		// terceira instancia = input + label
		linha = 2;
		inputs.putScalar(new int[] { linha, COL_NEURON_INPUT_1 }, 0);
		inputs.putScalar(new int[] { linha, COL_NEURON_INPUT_2 }, 1);

		labels.putScalar(new int[] { linha, CLASSE_F }, LABEL_V);
		labels.putScalar(new int[] { linha, CLASSE_V }, LABEL_F);

		// quarta instancia = input + label
		linha = 3;
		inputs.putScalar(new int[] { linha, COL_NEURON_INPUT_1 }, 1);
		inputs.putScalar(new int[] { linha, COL_NEURON_INPUT_2 }, 1);

		labels.putScalar(new int[] { linha, CLASSE_F }, LABEL_F);
		labels.putScalar(new int[] { linha, CLASSE_V }, LABEL_V);

		// criando o dataset
		this.ds = new DataSet(inputs, labels);

		System.out.println(ds);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see br.com.marketmining.dl4j.sample.Net#fit()
	 */
	@Override
	public void fit() {
		this.model = new MultiLayerNetwork(this.configuration);
		this.model.init();

		UIServer uiServer = UIServer.getInstance();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		uiServer.attach(statsStorage);
		this.model.setListeners(new StatsListener(statsStorage));

		this.model.fit(ds);

	}

}
