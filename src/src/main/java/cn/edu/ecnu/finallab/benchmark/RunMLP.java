package cn.edu.ecnu.finallab.benchmark;

import cn.edu.ecnu.finallab.MnistData;
import cn.edu.ecnu.finallab.benchmark.flink.MLPFlink;
import cn.edu.ecnu.finallab.benchmark.spark.MLPSpark;

import java.util.ArrayList;

public class RunMLP {
    public static double run(String[] args) throws Exception {
        final String imagePath = args[4];
        final String labelPath = args[5];
        int numHiddens = 20;
        double learningRate = 2;
        int epoch = Integer.parseInt(args[3]);
        int batchSize = 64;

        System.out.println("Running MLP on MNIST.");
        System.out.format("Num Epoch: %d, Batch size: %d, Num hiddens: %d, Learning rate: %.1f\n"
            , epoch, batchSize, numHiddens, learningRate);

        MnistData data = new MnistData(imagePath, labelPath);
        ArrayList<ArrayList<ArrayList<Double>>> imageLoader = new ArrayList<>();
        ArrayList<ArrayList<ArrayList<Double>>> labelLoader = new ArrayList<>();
        data.getAllBatch(batchSize, imageLoader, labelLoader);
        int batchNum = data.getBatchNum();

        double usedTime = 0.;

        if (args[1].equals("spark")) {
            System.out.println("Training with Spark...");
            long startTime = System.currentTimeMillis();

            MLPSpark.runSpark(args, data, imageLoader, labelLoader, numHiddens, learningRate, batchNum, epoch);

            long endTime = System.currentTimeMillis();
            usedTime = (endTime - startTime) * 1.0 / 1000;
        }
        else if (args[1].equals("flink")) {
            System.out.println("Training with Flink (without Bulk Iteration)...");
            long startTime = System.currentTimeMillis();

            MLPFlink.runFlink(args, data, imageLoader, labelLoader, numHiddens, learningRate, batchNum, epoch);

            long endTime = System.currentTimeMillis();
            usedTime = (endTime - startTime) * 1.0 / 1000;
        }
        else if (args[1].equals("flink-iterative")) {
            System.out.println("Training with Flink (with Bulk Iteration)...");
            System.out.println("[Warning] Training with Bulk Iteration will create many copies of the model instead of passing in references, so the results are not comparable.");
            long startTime = System.currentTimeMillis();

            MLPFlink.runFlinkBulk(args, data, imageLoader, labelLoader, numHiddens, learningRate, batchNum, epoch);

            long endTime = System.currentTimeMillis();
            usedTime = (endTime - startTime) * 1.0 / 1000;
        }

        return usedTime;
    }
}
