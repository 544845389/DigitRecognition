package com.example.dl4jdemo.controller;


import org.apache.commons.io.IOUtils;
import org.apache.commons.io.filefilter.FileFilterUtils;
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
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import tachyon.org.jets3t.service.model.MultipartPart;

import javax.imageio.ImageIO;
import javax.servlet.http.HttpServletRequest;
import javax.ws.rs.POST;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.UUID;

/**
 * 上传图片识别
 */
@Controller
public class UploadImageRecognition implements InitializingBean {


    @Value("${filePath}")
    private String path;


    private MultiLayerNetwork net;


    @Override
    public void afterPropertiesSet() throws Exception {
        net = ModelSerializer.restoreMultiLayerNetwork(new File(path + "\\model.zip"));

    }


    @ResponseBody
    @RequestMapping(value = "/upImage", method = RequestMethod.POST)
    public String fileUpload(@RequestParam(value = "file") MultipartFile file, HttpServletRequest request) throws IOException {
        if (file.isEmpty()) {
            System.out.println("文件为空空");
        }
        String fileName = file.getOriginalFilename();  // 文件名
        String suffixName = fileName.substring(fileName.lastIndexOf("."));  // 后缀名

        fileName = UUID.randomUUID() + suffixName; // 新文件名

        String filePath = path + "/" + fileName;
        File dest = new File(filePath);
        if (!dest.getParentFile().exists()) {
            dest.getParentFile().mkdirs();
        }
        file.transferTo(dest);

        filePath = zoomImage(filePath);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        ImageRecordReader testRR = new ImageRecordReader(28, 28, 1);
        File testData = new File(filePath);
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS);
        testRR.initialize(testSplit);

        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, 1);
        testIter.setPreProcessor(scaler);
        INDArray array = testIter.next().getFeatureMatrix();

        dest.delete();
        return String.valueOf(net.predict(array)[0]);

    }


    private String zoomImage(String filePath) {
        String imagePath = path + "/" + UUID.randomUUID().toString() + ".png";
        try {
            BufferedImage bufferedImage = ImageIO.read(new File(filePath));
            Image image = bufferedImage.getScaledInstance(28, 28, Image.SCALE_SMOOTH);
            BufferedImage tag = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
            Graphics g = tag.getGraphics();
            g.drawImage(image, 0, 0, null);
            g.dispose();
            ImageIO.write(tag, "png", new File(imagePath));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return imagePath;
    }


}
