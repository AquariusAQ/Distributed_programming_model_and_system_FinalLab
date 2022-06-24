package cn.edu.ecnu.finallab;

import com.csvreader.CsvReader;

import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;

public class MnistData {
    private ArrayList<ArrayList<Double>> images;
    private ArrayList<ArrayList<Double>> labels;
    private Integer numFeatures;
    private Integer numSamples;
    private Integer sizeBatch = -1;
    private Integer numBatch = -1;


    public MnistData() {
        System.out.println("Input data is empty and this object is read-only.");
        new MnistData(new ArrayList<>(), new ArrayList<>());
    }

    public MnistData(String imagePath, String labelPath) {
        ArrayList<ArrayList<Double>> imageList = new ArrayList<>();
        ArrayList<ArrayList<Double>> labelList = new ArrayList<>();
        try {
            CsvReader reader = new CsvReader(imagePath, ',', Charset.forName("UTF-8"));
            while (reader.readRecord()) {
                ArrayList<Double> arr = new ArrayList<>();
                for (String str : Arrays.asList(reader.getValues())) {
                    arr.add(Double.parseDouble(str));
                }
                imageList.add(arr);
            }
            reader.close();

            CsvReader reader2 = new CsvReader(labelPath, ',', Charset.forName("UTF-8"));
            while (reader2.readRecord()) {
                ArrayList<Double> arr = new ArrayList<>();
                for (String str : Arrays.asList(reader2.getValues())) {
                    arr.add(Double.parseDouble(str));
                }
                labelList.add(arr);
            }
            reader2.close();
            images = imageList;
            labels = labelList;
            if (images.size() != labels.size()) {
                System.out.println("Data Length should be same!");
                System.exit(-1);
            }
            numSamples = images.size();
            numFeatures = images.get(0).size();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public MnistData(ArrayList<ArrayList<Double>> imgs, ArrayList<ArrayList<Double>> y) {
        if (imgs.size() != y.size()) {
            System.out.println("Data Length should be same!");
            System.exit(-1);
        }
        images = imgs;
        labels = y;
        numSamples = imgs.size();
        numFeatures = imgs.get(0).size();
    }

    public void getAllBatch(int batchSize, ArrayList<ArrayList<ArrayList<Double>>> imageLoader, ArrayList<ArrayList<ArrayList<Double>>> labelLoader) {
        int batchNum =  (numSamples - numSamples % batchSize) / batchSize;
        sizeBatch = batchSize;
        numBatch = batchNum;

        for (int i = 0; i < batchNum; i++) {
            imageLoader.add(new ArrayList<>(images.subList(i * batchSize, (i + 1) * batchSize)));
            labelLoader.add(new ArrayList<>(labels.subList(i * batchSize, (i + 1) * batchSize)));
        }

        return;
    }

    public ArrayList<ArrayList<Double>> getImages() {
        return images;
    }

    public ArrayList<ArrayList<Double>> getLabels() {
        return labels;
    }

    public Integer getNumFeatures() {
        return numFeatures;
    }

    public Integer getNumSamples() {
        return numSamples;
    }

    public ArrayList<Double> getImageByIndex(int index) {
        return images.get(index);
    }

    public ArrayList<Double> getLabelByIndex(int index) {
        return labels.get(index);
    }

    public int getLiteralLabelByIndex(int index) {
        ArrayList<Double> label = labels.get(index);
        for (int i = 0; i < label.size(); i++) {
            if (Math.abs(label.get(i) - 1.) < 0.01) {
                return i;
            }
        }
        System.out.println("Cannot find label!");
        return -1;
    }

    public Integer getBatchNum() { return numBatch; }

    public Integer getBatchSize() { return sizeBatch; }
}
