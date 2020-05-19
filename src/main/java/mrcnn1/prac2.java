package mrcnn1;
import org.apache.log4j.BasicConfigurator;

import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.github.sh0nk.matplotlib4j.Plot;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import java.io.IOException;


public class prac2 {
	public static void main(String args[]) {
		Plot plt = Plot.create();
		BasicConfigurator.configure();
		//Building the output layer
		OutputLayer outputlayer = new OutputLayer.Builder()
		    .nIn(784) //The number of inputs feed from the input layer
		    .nOut(10) //The number of output values the output layer is supposed to take
		    .weightInit(WeightInit.XAVIER) //The algorithm to use for weights initialization
		    .activation(Activation.SOFTMAX) //Softmax activate converts the output layer into a probability distribution
		    .build(); //Building our output layer
		//Since this is a simple network with a stack of layers we're going to configure a MultiLayerNetwork
		MultiLayerConfiguration logisticRegressionConf= new NeuralNetConfiguration.Builder()
		    //High Level Configuration
		    .seed(123)
		    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		    .updater(new Nesterovs(0.1, 0.9)) 
		    //For configuring MultiLayerNetwork we call the list method
		    .list() 
		    .layer(0, outputlayer) //    <----- output layer fed here
		    .build() ;//Building Configuration
		
		
		MultiLayerNetwork model = new MultiLayerNetwork(logisticRegressionConf);
		model.init();
		model.fit();
		//print the score with every 10 iteration
		model.setListeners(new ScoreIterationListener(10));
		
	}
}
