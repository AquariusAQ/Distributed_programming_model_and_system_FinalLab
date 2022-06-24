package cn.edu.ecnu.finallab.benchmark.spark;

import Jama.Matrix;
import cn.edu.ecnu.finallab.benchmark.RunKmeans;
import cn.edu.ecnu.finallab.model.Utils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;

public class PCASpark {
    public static void runSpark(String[] args, ArrayList<Matrix> imageList, int epoch) throws Exception {
        SparkSession spark = SparkSession
                .builder()
                // .master("local")
                .appName("PCA")
                .getOrCreate();

        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("ERROR");
        final int width = imageList.get(0).getColumnDimension();

        ArrayList<Tuple3<Integer, Matrix, Double>> dataList = new ArrayList<>();
        for (Matrix img : imageList) {
            dataList.add(new Tuple3<>(0, img, 0.));
        }

        JavaRDD<Tuple3<Integer, Matrix, Double>> dataRDD = sc.parallelize(dataList);

        if (args[2].equals("vanilla")) {

            for (int i = 0; i < epoch; i++) {
                dataRDD = dataRDD
                        .map((data) -> {
                            Matrix lastImage = data._2();
                            int component = width - data._1() % 10 - 11;
                            Matrix newImage = Utils.PCA(lastImage, component);
                            Double rmse = Utils.RMSE(lastImage, newImage);
//                        for (int j = 0; j < 100; j++) {
//                            newImage = Utils.PCA(lastImage, component);
//                            rmse = Utils.RMSE(lastImage, newImage);
//                        }
                            return new Tuple3<>(data._1()+1, newImage, rmse);
                        });

//            double RMSESum = dataRDD
//                    .map((data) -> data._3())
//                    .reduce((data1, data2) -> data1 + data2);
//
//            System.out.format("Epoch %d, Component %d, RMSE %.12f\n", i, width-1-i, RMSESum / imageList.size());
            }

            double RMSESum = dataRDD
                    .map((data) -> data._3())
                    .reduce((data1, data2) -> data1 + data2);

            System.out.format("Final RMSE %.12f\n", RMSESum / imageList.size());
        }
        else if (args[2].equals("communication")) {

            for (int i = 0; i < epoch; i++) {
                dataRDD = dataRDD
                        .map((data) -> {
                            Matrix lastImage = data._2();
                            int component = width - data._1() % 10 - 11;
                            Matrix newImage = Utils.PCA(lastImage, component);
                            Double rmse = Utils.RMSE(lastImage, newImage);
//                        for (int j = 0; j < 100; j++) {
//                            newImage = Utils.PCA(lastImage, component);
//                            rmse = Utils.RMSE(lastImage, newImage);
//                        }
                            return new Tuple3<>(data._1() + 1, newImage, rmse);
                        });

            double RMSESum = dataRDD
                    .map((data) -> data._3())
                    .reduce((data1, data2) -> data1 + data2);

            System.out.format("RMSE %.12f\n", RMSESum / imageList.size());
            }

            double RMSESum = dataRDD
                    .map((data) -> data._3())
                    .reduce((data1, data2) -> data1 + data2);

            System.out.format("Final RMSE %.12f\n", RMSESum / imageList.size());
        }
        else {
            System.out.println("Unknown job type");
        }
    }
}
