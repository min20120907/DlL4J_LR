package mrcnn1;

import java.awt.BasicStroke;
import java.awt.Color;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.DatasetRenderingOrder;
import org.jfree.chart.plot.Marker;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYDotRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import au.com.bytecode.opencsv.CSVReader;
import sklearn.preprocessing.MinMaxScaler;

public class prac3 {
	public static final XYSeriesCollection dataset = new XYSeriesCollection( );
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		BasicConfigurator.configure();
		List<Float> smoker_bmi = new ArrayList<Float>();
		List<Float> smoker_charges= new ArrayList<Float>();
		List<Float> non_smoker_bmi= new ArrayList<Float>();
		List<Float> non_smoker_charges= new ArrayList<Float>();
		
		File file = new File("/home/min20120907/eclipse-workspace/mrcnn1/src/main/java/mrcnn1/insurance.csv");
		FileReader fReader = new FileReader(file);
		CSVReader csvReader = new CSVReader(fReader);
		
		
		List<String[]> list = csvReader.readAll();
		csvReader.close();
		for(String[] ss : list){
			try {
				float bmi = Float.parseFloat(ss[2]);
				float charges = Float.parseFloat(ss[6]);
				if(ss[4].equals("yes")) {
					smoker_bmi.add(bmi);
					smoker_charges.add(charges);
				}else {
					non_smoker_bmi.add(bmi);
					non_smoker_charges.add(charges);
				}
				 
			}catch(RuntimeException re) {
				System.err.println("Can not parse!");
			}
		}
		//Scaling
		ArrayList<Float> new_x1 = MinMaxScaler_fit_transform(smoker_bmi);
		ArrayList<Float> new_y1 = MinMaxScaler_fit_transform(smoker_charges);
		ArrayList<Float> new_x2 = MinMaxScaler_fit_transform(non_smoker_bmi);
		ArrayList<Float> new_y2 = MinMaxScaler_fit_transform(non_smoker_charges);
		//array converting
		INDArray new_x1_n = Nd4j.create(MinMaxScaler_fit_transform(smoker_bmi)).reshape(new int[]{new_x1.size(),1});
		INDArray new_y1_n = Nd4j.create(MinMaxScaler_fit_transform(smoker_charges)).reshape(new int[]{new_y1.size(),1});
		INDArray new_x2_n = Nd4j.create(MinMaxScaler_fit_transform(non_smoker_bmi)).reshape(new int[]{new_x2.size(),1});
		INDArray new_y2_n = Nd4j.create(MinMaxScaler_fit_transform(non_smoker_charges)).reshape(new int[]{new_y2.size(),1});
		//Datasets pairing
		DataSet set1 = new DataSet(new_x1_n, new_y1_n);
		DataSet set2 = new DataSet(new_x2_n, new_y2_n);
		//smoker model
		   int seed = 123;
	       double learningRate = 0.001;
	       int numInputs = 1;
	       int numOutputs = 1;
	       int numEpoches = 20;
	       MultiLayerConfiguration conf_smoker = ( new NeuralNetConfiguration.Builder()
	               .seed(seed)
	               .weightInit(WeightInit.XAVIER)
	               .updater(new Adam(learningRate))
	               .list())
	               .layer(new OutputLayer.Builder(LossFunction.MSE)
	            		   .nIn(numInputs).nOut(numOutputs)
	                       .activation(Activation.IDENTITY)
	                       .build()
	                       )
	               .build();
	    MultiLayerNetwork model_smoker = new MultiLayerNetwork(conf_smoker);
	    model_smoker.setEpochCount(numEpoches);
		model_smoker.init();
	    model_smoker.setListeners(new ScoreIterationListener(1));
	    
	    //non_smoker model
	    int seed1 = 123;
	       double learningRate1 = 0.001;
	       int numInputs1 = 1;
	       int numOutputs1 = 1;
	       int numEpoches1 = 20;
	       MultiLayerConfiguration conf_non_smoker = ( new NeuralNetConfiguration.Builder()
	               .seed(seed1)
	               .weightInit(WeightInit.XAVIER)
	               .updater(new Adam(learningRate1))
	               .list())
	               .layer(new OutputLayer.Builder(LossFunction.MSE)
	            		   .nIn(numInputs1).nOut(numOutputs1)
	                       .activation(Activation.IDENTITY)
	                       .build()
	                       )
	               .build();
	    MultiLayerNetwork model_non_smoker = new MultiLayerNetwork(conf_non_smoker);
	    model_non_smoker.setEpochCount(numEpoches1);
		model_non_smoker.init();
	    //model_non_smoker.setListeners(new ScoreIterationListener(1));
	    
	    INDArray a = Nd4j.concat(1,new_x1_n, new_y1_n);
	    INDArray b = Nd4j.concat(1, new_x2_n, new_y2_n);
	    
	    for(int i = 0;i<100;i++) {
	    	model_smoker.fit(set1);
	    	model_non_smoker.fit(set1);
	    	
	    }
	    final XYSeries set1_series = new XYSeries( "Smoker" );
	    final XYSeries set2_series = new XYSeries( "Non-Smoker" );
	    final XYSeries lr1_series = new XYSeries( "Linear Regression 1" );
	    final XYSeries lr2_series = new XYSeries( "Linear Regression 2" );
	    
	    for(int j =0;j<new_x1.size();j++) {
	    	set1_series.add(new_x1.get(j),new_y1.get(j));
	    	set2_series.add(new_x2.get(j),new_y2.get(j));
	    }
	    
	    dataset.addSeries( set1_series );
	    dataset.addSeries( set2_series );
	    
	    XYScatterChart_AWT chart = new XYScatterChart_AWT("Result Iteration = 1",
	            "Linear Regression With NeuralNetwork");
	         chart.pack( );
	         
	         RefineryUtilities.centerFrameOnScreen( chart );          
	         chart.setVisible( true ); 
    	for(double j =0;j<1;j+=0.1) {
	    	lr1_series.add(model_smoker.getLayer(0).score(),j);
	    	lr2_series.add(model_non_smoker.getLayer(0).score(),j);
	    }
    	dataset.addSeries(lr1_series);
    	dataset.addSeries(lr2_series);
	}

	//MinMaxScaler
	public static ArrayList<Float> MinMaxScaler_fit_transform(List<Float> input){
		ArrayList<Float> tmp =(ArrayList<Float>) input;
		float max = tmp.get(0), min = tmp.get(0);
		for(float i : tmp) {
			if(max < i)
				max =i;
			if(min > i)
				min = i;
		}
		for(int i = 0;i<input.size();i++) {
			tmp.set(i, (input.get(i)-min) / (max-min));
		}
		return tmp;
	}
}
class XYScatterChart_AWT extends ApplicationFrame {

	   public XYScatterChart_AWT( String applicationTitle, String chartTitle ) {
	      super(applicationTitle);
	      JFreeChart xyscatterChart = ChartFactory.createScatterPlot(
	         chartTitle ,
	         "BMI(Normalized)" ,
	         "Charges(Normalized)" ,
	         createDataset() ,
	         PlotOrientation.VERTICAL, true, true, true);
	      
	      ChartPanel chartPanel = new ChartPanel( xyscatterChart );
	      chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 367 ) );
	      final XYPlot plot = xyscatterChart.getXYPlot();
	      XYLineAndShapeRenderer r = new XYLineAndShapeRenderer(true, false);
	      r.setSeriesPaint(2, Color.blue);
	      r.setSeriesPaint(3, Color.orange);
	      r.setSeriesVisible(1, false);
	      r.setSeriesVisible(0, false);
	      plot.mapDatasetToDomainAxis(0, 0);
	      plot.mapDatasetToRangeAxis(0, 0);
	      XYDotRenderer renderer = new XYDotRenderer( );
	      renderer.setSeriesPaint( 0 , Color.RED );
	      renderer.setSeriesPaint( 1 , Color.GREEN );
	      renderer.setDotHeight(3);
	      renderer.setDotWidth(3);
	      renderer.setSeriesVisible(2, false);
	      renderer.setSeriesVisible(3, false);
	      plot.setRenderer(0, renderer);
	      plot.setRenderer(1, r);
	      plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD);
	      plot.mapDatasetToDomainAxis(1, 0);
	      plot.mapDatasetToRangeAxis(1, 1);
	      setContentPane( chartPanel ); 
	   }
	   
	   private XYDataset createDataset( ) {
	      
	      return prac3.dataset;
	   }
}