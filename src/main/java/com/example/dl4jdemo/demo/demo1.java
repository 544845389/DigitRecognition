package com.example.dl4jdemo.demo;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class demo1 {


    /**
     * 手写字体识别
     * @param args
     * @throws Exception
     */


    public static void main(String[] args) throws Exception {

        int nChannels = 1;      //black & white picture, 3 if color image
        int nEpochs = 10;       //total rounds of training
        int batchSize = 64; // batch size for each epoch
        int outputNum = 10; // 分类的个数
        int iterations = 1;     //number of iteration in each traning round
        int seed = 123;         //初始化权重的随机种子

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize , true , 123456);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize , false , 123456);


        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
                .cnnInputSize(28, 28, 1);
                // The builder needs the dimensions of the image along with the number of channels. these are 28x28 images in one channel
                //new ConvolutionLayerSetup(builder,28,28,1);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));
        // a listener which can print loss function score after each iteration
//
        for( int i = 0; i < nEpochs; ++i ) {
            model.fit(mnistTrain);
            System.out.println("*** Completed epoch " + i + "***");

            System.out.println("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while(mnistTest.hasNext()){
                DataSet ds = mnistTest.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }
            System.out.println(eval.stats());
            mnistTest.reset();
        }


//        System.out.println("Train model....");
//        for( int i=0; i<nEpochs; i++ ){
//            System.out.println("Epoch " + i);
//            model.fit(mnistTrain);
//        }

        System.out.println("保存开始");

        File saveModel = new File("C:\\Users\\54484\\Desktop\\deeplearning4j\\model\\model.zip");

        boolean saveOrUpdate = false;

        ModelSerializer.writeModel(model , saveModel , saveOrUpdate);

        System.out.println("保存结束");
    }

}
