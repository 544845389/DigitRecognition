package com.example.dl4jdemo.controller;


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
import org.springframework.beans.factory.InitializingBean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import sun.misc.BASE64Decoder;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.UUID;

@Controller
public class HandwritingController implements InitializingBean {


    @Value("${filePath}")
    private String   path;


    private MultiLayerNetwork net;


    @Override
    public void afterPropertiesSet() throws Exception {

        net = ModelSerializer.restoreMultiLayerNetwork(new File("C:\\Users\\54484\\Desktop\\deeplearning4j\\model\\model.zip"));

    }


    /**
     *  手写字体识别
     * @param
     * @return
     */
    @ResponseBody
    @RequestMapping(value = "/Identify", method = RequestMethod.GET)
    public int Identify(@RequestParam(value = "img") String img) throws IOException {
       String imagePath =  generateImage(img);
       imagePath= zoomImage(imagePath);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        ImageRecordReader testRR = new ImageRecordReader(28, 28, 1);
        File testData = new File(imagePath);
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS);
        testRR.initialize(testSplit);

        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, 1);
        testIter.setPreProcessor(scaler);
        INDArray array = testIter.next().getFeatureMatrix();
        return net.predict(array)[0];
    }


    private String generateImage(String img) {
        BASE64Decoder decoder = new BASE64Decoder();
        String filePath = path +"/"+ UUID.randomUUID().toString()+".png";
        try {
            byte[] b = decoder.decodeBuffer(img);
            for (int i = 0; i < b.length; ++i) {
                if (b[i] < 0) {
                    b[i] += 256;
                }
            }
            OutputStream out = new FileOutputStream(filePath);
            out.write(b);
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return filePath;
    }

    private String zoomImage(String filePath){
        String imagePath = path +"/"+ UUID.randomUUID().toString()+".png";
        try {
            BufferedImage bufferedImage = ImageIO.read(new File(filePath));
            Image image = bufferedImage.getScaledInstance(28, 28, Image.SCALE_SMOOTH);
            BufferedImage tag = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            Graphics g = tag.getGraphics();
            g.drawImage(image, 0, 0, null);
            g.dispose();
            ImageIO.write(tag, "png",new File(imagePath));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return imagePath;
    }


}
