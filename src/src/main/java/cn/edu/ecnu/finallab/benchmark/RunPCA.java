package cn.edu.ecnu.finallab.benchmark;

import Jama.Matrix;
import cn.edu.ecnu.finallab.MnistData;
import cn.edu.ecnu.finallab.benchmark.flink.PCAFlink;
import cn.edu.ecnu.finallab.benchmark.spark.PCASpark;
import cn.edu.ecnu.finallab.model.Utils;

import java.util.ArrayList;

public class RunPCA {
    public static double run(String[] args) throws Exception {
        final String imagePath = args[4];
        final String labelPath = args[5];

        int epoch = Integer.parseInt(args[3]);

        System.out.println("Running PCA on MNIST.");
        System.out.format("Num Epoch: %d\n", epoch);

        MnistData data = new MnistData(imagePath, labelPath);
        ArrayList<ArrayList<Double>> imageFlat = data.getImages();
        int width = (int)Math.sqrt(data.getNumFeatures());
        ArrayList<Matrix> images = new ArrayList<>();
        for (ArrayList<Double> imgFlat : imageFlat) {
            double[][] img = new double[width][width];
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < width; j++) {
                    img[i][j] = imgFlat.get(i * width + j);
                }
            }
            images.add(new Matrix(Utils.standardization(img)));
        }

        double usedTime = 0.;

        if (args[1].equals("spark")) {
            System.out.println("Training with Spark...");

            long startTime = System.currentTimeMillis();

            PCASpark.runSpark(args, images, epoch);

            long endTime = System.currentTimeMillis();
            usedTime = (endTime - startTime) * 1.0 / 1000;
        }
        else if (args[1].equals("flink")) {
            System.out.println("Training with Flink (without Bulk Iteration)...");

            long startTime = System.currentTimeMillis();

            PCAFlink.runFlink(args, images, epoch);

            long endTime = System.currentTimeMillis();
            usedTime = (endTime - startTime) * 1.0 / 1000;
        }
        else if (args[1].equals("flink-bulk")) {
            System.out.println("Training with Flink (with Bulk Iteration)...");

            long startTime = System.currentTimeMillis();

            PCAFlink.runFlinkBulk(args, images, epoch);

            long endTime = System.currentTimeMillis();
            usedTime = (endTime - startTime) * 1.0 / 1000;
        }

        return usedTime;
    }
}
