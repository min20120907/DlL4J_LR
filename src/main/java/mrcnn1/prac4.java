package mrcnn1;



import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;

/**
* "Linear" Data Classification Example
*
* Based on the data from Jason Baldridge:
* https://github.com/jasonbaldridge/try-tf/tree/master/simdata
*
* @author Josh Patterson
* @author Alex Black (added plots)
*
*/
@SuppressWarnings("DuplicatedCode")
public class prac4 {

   public static String dataLocalPath;

   public static void main(String[] args) throws Exception {
	   BasicConfigurator.configure();
	   int seed = 123;
       double learningRate = 0.001;
       int numInputs = 1;
       int numOutputs = 1;

       MultiLayerConfiguration conf_smoker = new NeuralNetConfiguration.Builder()
               .seed(seed)
               .weightInit(WeightInit.XAVIER)
               .updater(new Adam(learningRate))
               .list()
               .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numOutputs)
                       .activation(Activation.IDENTITY)
                       .build())
               .build();
       MultiLayerNetwork model = new MultiLayerNetwork(conf_smoker);
       model.init();
       model.setListeners(new ScoreIterationListener(100));
       for( int i=0; i<50; i++ ) {
           model.fit();
       }


   }
}
