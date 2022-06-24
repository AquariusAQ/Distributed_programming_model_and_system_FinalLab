package cn.edu.ecnu.finallab.benchmark.spark;

import cn.edu.ecnu.finallab.MnistData;
import cn.edu.ecnu.finallab.model.MLP;
import cn.edu.ecnu.finallab.model.Utils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.ArrayList;

public class MLPSpark {
    public static void runSpark(String[] args, MnistData data, ArrayList<ArrayList<ArrayList<Double>>> imageLoader,
                                ArrayList<ArrayList<ArrayList<Double>>> labelLoader,
                                int numHiddens, double learningRate, int batchNum, int epoch) throws Exception {
        SparkSession spark = SparkSession
                .builder()
                // .master("local")
                .appName("MLP")
                .getOrCreate();

        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("ERROR");

        MLP mlp = new MLP(data.getNumFeatures(), 10, numHiddens, learningRate);

        ArrayList<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>> dataTuple = new ArrayList<>();
        for (int i = 0; i < batchNum; i++) {
            ArrayList<ArrayList<ArrayList<Double>>> sampleData = new ArrayList<>();
            sampleData.add(imageLoader.get(i));
            sampleData.add(labelLoader.get(i));
            sampleData.add(new ArrayList<>());
            Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> x =
                    new Tuple2<>(mlp, sampleData);
            dataTuple.add(x);
        }
        JavaRDD<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>> dataRDD = sc.parallelize(dataTuple);


        for (int i = 0; i < epoch; i++) {
            dataRDD = dataRDD.map((batch) -> {
                ArrayList<ArrayList<Double>> pred_y = batch._1.forward(batch._2.get(0));
                batch._1.backward(batch._2.get(1));
                batch._1.update();
                ArrayList<ArrayList<ArrayList<Double>>> sampleData = batch._2;
                sampleData.set(2, pred_y);
                return new Tuple2<>(batch._1(), sampleData);
            });

            if (args[2].equals("communication")) {
                Tuple2<Double, Double> result = dataRDD
                        .map((batch) -> {
                            ArrayList<ArrayList<Double>> pred_y = batch._2.get(2);
                            double loss = Utils.computeLoss(pred_y, batch._2.get(1));
                            double accuracy = Utils.computeAccuracy(pred_y, batch._2.get(1));
                            return new Tuple2<>(loss, accuracy);
                        }).reduce((r1, r2) -> new Tuple2<>(r1._1+r2._1, r1._2+r2._2));
//            System.out.println("Result: " + resultRDD);
                double loss = result._1 / batchNum;
                double accuracy = result._2 / batchNum;
                System.out.format("\tEpoch: %d, Loss: %.4f, Acc: %.4f\n", i+1, loss, accuracy);
            }
        }

        if (args[2].equals("vanilla")) {
            Tuple2<Double, Double> result = dataRDD
                    .map((batch) -> {
                        ArrayList<ArrayList<Double>> pred_y = batch._2.get(2);
                        double loss = Utils.computeLoss(pred_y, batch._2.get(1));
                        double accuracy = Utils.computeAccuracy(pred_y, batch._2.get(1));
                        return new Tuple2<>(loss, accuracy);
                    }).reduce((r1, r2) -> new Tuple2<>(r1._1+r2._1, r1._2+r2._2));
//            System.out.println("Result: " + resultRDD);
            double loss = result._1 / batchNum;
            double accuracy = result._2 / batchNum;
            System.out.format("\tFinal Loss: %.4f, Acc: %.4f\n", loss, accuracy);
        }
    }
}
