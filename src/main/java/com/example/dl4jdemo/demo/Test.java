package com.example.dl4jdemo.demo;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.UUID;

public class Test {


    public static void main(String[] args) throws IOException {

            test();

//        String url = "F:\\github\\deeplearning4j\\dl4jdemo\\d51b23e5-5eeb-45e3-b618-d4664073a42d.png";
//        zoomImage(url);

    }

    private static void test() throws IOException {
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(new File("C:\\Users\\54484\\Desktop\\deeplearning4j\\model\\model.zip"));

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        ImageRecordReader testRR = new ImageRecordReader(28, 28, 1);

        File testData = new File("F:\\github\\deeplearning4j\\dl4jdemo\\test.jpg");
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS);
        testRR.initialize(testSplit);

        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, 1);
        testIter.setPreProcessor(scaler);
        INDArray array = testIter.next().getFeatureMatrix();
        System.out.println(net.predict(array)[0]);
    }


    private static String zoomImage(String filePath){
        String imagePath = "F:\\github\\deeplearning4j\\dl4jdemo\\test.jpg";
        try {
            BufferedImage bufferedImage = ImageIO.read(new File(filePath));
            Image image = bufferedImage.getScaledInstance(28, 28, Image.SCALE_SMOOTH);
            BufferedImage tag = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            Graphics g = tag.getGraphics();
            g.drawImage(image, 0, 0, null);
            g.dispose();
            ImageIO.write(tag, "jpg",new File(imagePath));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return imagePath;
    }



}
