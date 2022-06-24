package cn.edu.ecnu.finallab.benchmark.flink;

import Jama.Matrix;
import cn.edu.ecnu.finallab.model.Utils;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.IterativeDataSet;
import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.util.List;

public class PCAFlink {
    public static void runFlink(String[] args, ArrayList<Matrix> imageList, int epoch) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        int width = imageList.get(0).getColumnDimension();

        ArrayList<Tuple3<Integer, Matrix, Double>> dataList = new ArrayList<>();
        for (Matrix img : imageList) {
            dataList.add(new Tuple3<>(width, img, 0.));
        }

        DataSet<Tuple3<Integer, Matrix, Double>> dataSet = env.fromCollection(dataList);

        if (args[2].equals("vanilla")) {
            for (int i = 0; i < epoch; i++) {
                dataSet = dataSet
                        .map(new MapFunction<Tuple3<Integer, Matrix, Double>, Tuple3<Integer, Matrix, Double>>() {
                            @Override
                            public Tuple3<Integer, Matrix, Double> map(Tuple3<Integer, Matrix, Double> data) throws Exception {
                                Matrix lastImage = data._2();
                                int component = width - data._1() % 10 - 11;
                                Matrix newImage = Utils.PCA(lastImage, component);
                                Double rmse = Utils.RMSE(lastImage, newImage);
                                return new Tuple3<>(data._1()+1, data._2(), rmse);
                            }
                        });
            }


            List<Double> RMSESumList = dataSet
                    .map((data) -> data._3())
                    .reduce((data1, data2) -> data1 + data2).collect();

            double RMSESum = RMSESumList.get(0);

            System.out.format("Final RMSE %.12f\n", RMSESum / imageList.size());
        }
        else if (args[2].equals("communication")) {
            for (int i = 0; i < epoch; i++) {
                dataSet = dataSet
                        .map(new MapFunction<Tuple3<Integer, Matrix, Double>, Tuple3<Integer, Matrix, Double>>() {
                            @Override
                            public Tuple3<Integer, Matrix, Double> map(Tuple3<Integer, Matrix, Double> data) throws Exception {
                                Matrix lastImage = data._2();
                                int component = width - data._1() % 10 - 11;
                                Matrix newImage = Utils.PCA(lastImage, component);
                                Double rmse = Utils.RMSE(lastImage, newImage);
                                return new Tuple3<>(data._1()+1, data._2(), rmse);
                            }
                        });

                DataSet<Double> RMSESumList = dataSet
                        .map((data) -> data._3())
                        .reduce((data1, data2) -> data1 + data2)
                        .map(RMSESum -> {
                            System.out.format("RMSE %.12f\n", RMSESum / imageList.size());
                            return RMSESum;
                        });

                if (RMSESumList.count() == 0) {
                    System.out.println("Unknown Error!");
                    break;
                }
            }


            List<Double> RMSESumList = dataSet
                    .map((data) -> data._3())
                    .reduce((data1, data2) -> data1 + data2).collect();

            double RMSESum = RMSESumList.get(0);

            System.out.format("Final RMSE %.12f\n", RMSESum / imageList.size());
        }
        else {
            System.out.println("Unknown job type");
        }

    }

    public static void runFlinkBulk(String[] args, ArrayList<Matrix> imageList, int epoch) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        int width = imageList.get(0).getColumnDimension();

        ArrayList<Tuple3<Integer, Matrix, Double>> dataList = new ArrayList<>();
        for (Matrix img : imageList) {
            dataList.add(new Tuple3<>(width, img, 0.));
        }

        DataSet<Tuple3<Integer, Matrix, Double>> dataSet = env.fromCollection(dataList);

        if (args[2].equals("vanilla")) {
            IterativeDataSet<Tuple3<Integer, Matrix, Double>> iterate = dataSet.iterate(epoch);

            DataSet<Tuple3<Integer, Matrix, Double>> iterateBody = iterate
                    .map(new MapFunction<Tuple3<Integer, Matrix, Double>, Tuple3<Integer, Matrix, Double>>() {
                        @Override
                        public Tuple3<Integer, Matrix, Double> map(Tuple3<Integer, Matrix, Double> data) throws Exception {
                            Matrix lastImage = data._2();
                            int component = width - data._1() % 10 - 11;
                            Matrix newImage = Utils.PCA(lastImage, component);
                            Double rmse = Utils.RMSE(lastImage, newImage);
                            return new Tuple3<>(data._1()+1, data._2(), rmse);
                        }
                    });

            DataSet<Tuple3<Integer, Matrix, Double>> finalData = iterate.closeWith(iterateBody);

            List<Double> RMSE = finalData
                    .map((data) -> data._3())
                    .reduce((data1, data2) -> data1 + data2).collect();

            double RMSESum = RMSE.get(0);

            System.out.format("Final RMSE %.12f\n", RMSESum / imageList.size());
        }
        else if (args[2].equals("communication")) {
            IterativeDataSet<Tuple3<Integer, Matrix, Double>> iterate = dataSet.iterate(epoch);

            DataSet<Tuple3<Integer, Matrix, Double>> iterateBody = iterate
                    .map(new MapFunction<Tuple3<Integer, Matrix, Double>, Tuple3<Integer, Matrix, Double>>() {
                        @Override
                        public Tuple3<Integer, Matrix, Double> map(Tuple3<Integer, Matrix, Double> data) throws Exception {
                            Matrix lastImage = data._2();
                            int component = width - data._1() % 10 - 11;
                            Matrix newImage = Utils.PCA(lastImage, component);
                            Double rmse = Utils.RMSE(lastImage, newImage);
                            return new Tuple3<>(data._1()+1, data._2(), rmse);
                        }
                    });

            DataSet<Double> RMSESumList = iterateBody
                    .map((data) -> data._3())
                    .reduce((data1, data2) -> data1 + data2)
                    .map(RMSESum -> {
                        System.out.format("RMSE %.12f\n", RMSESum / imageList.size());
                        return RMSESum;
                    });

            DataSet<Tuple3<Integer, Matrix, Double>> finalData = iterate.closeWith(iterateBody, RMSESumList);

            List<Double> RMSE = finalData
                    .map((data) -> data._3())
                    .reduce((data1, data2) -> data1 + data2).collect();

            double RMSESum = RMSE.get(0);

            System.out.format("Final RMSE %.12f\n", RMSESum / imageList.size());
        }
        else {
            System.out.println("Unknown job type");
        }


    }
}
