package cn.edu.ecnu.finallab.benchmark.spark;

import cn.edu.ecnu.finallab.Run;
import cn.edu.ecnu.finallab.benchmark.RunKmeans;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.util.List;

public class KmeansSpark {
    public static void runSpark(String[] args, ArrayList<RunKmeans.Centroid> centroidList,
                                ArrayList<RunKmeans.Point> pointList, int epoch) throws Exception {
        SparkSession spark = SparkSession
                .builder()
                // .master("local")
                .appName("Kmeans")
                .getOrCreate();

        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("ERROR");


        if (args[2].equals("vanilla")) {
            JavaRDD<RunKmeans.Point> pointRDD = sc.parallelize(pointList);
            JavaRDD<RunKmeans.Centroid> centroidRDD = sc.parallelize(centroidList);

            for (int i = 0; i < epoch; i++) {
                List<RunKmeans.Centroid> centroids = centroidRDD.collect();
                Broadcast<List<RunKmeans.Centroid>> centroidBroadcast = sc.broadcast(centroids);

                centroidRDD = pointRDD
                        // 寻找最近的聚类中心
                        .map((RunKmeans.Point p) -> {
                            double minDistance = Double.MAX_VALUE;
                            int closestCentroidId = -1;
                            // 检查所有的聚类中心
                            for (RunKmeans.Centroid centroid : centroidBroadcast.getValue()) {
                                // 计算每个点与聚类中心的距离（欧式距离）
                                double distance = p.euclideanDistance(centroid);

                                // 满足条件更新最近的聚类中心Id
                                if (distance < minDistance) {
                                    minDistance = distance;
                                    closestCentroidId = centroid.id;
                                }
                            }

                            // 生成一个包含聚类中心id和数据点的元组tuple.
                            return new Tuple2<>(closestCentroidId, p);
                        })
                        .map(t -> new Tuple3<>(t._1, t._2, 1L))
                        .keyBy(t -> t._1())
                        // 求每一类点的坐标和
                        .reduceByKey((val1, val2) -> new Tuple3<>(val1._1(), val1._2().add(val2._2()), val1._3() + val2._3()))
                        // 求新的聚类中心
                        .map((value) -> new RunKmeans.Centroid(value._2._1(), value._2._2().div(value._2._3())));
            }

            List<RunKmeans.Centroid> centroids = centroidRDD.collect();
            Broadcast<List<RunKmeans.Centroid>> centroidBroadcast = sc.broadcast(centroids);
            int[][] finalResult = pointRDD
                    .map((RunKmeans.Point p) -> {
                        double minDistance = Double.MAX_VALUE;
                        int closestCentroidId = -1;

                        // 检查所有的聚类中心
                        for (RunKmeans.Centroid centroid : centroidBroadcast.getValue()) {
                            // 计算每个点与聚类中心的距离（欧式距离）
                            double distance = p.euclideanDistance(centroid);

                            // 满足条件更新最近的聚类中心Id
                            if (distance < minDistance) {
                                minDistance = distance;
                                closestCentroidId = centroid.id;
                            }
                        }

                        // 生成一个包含聚类中心id和数据点的元组tuple.
                        return new Tuple2<>(closestCentroidId, p);
                    })
                    .map(value -> {
                        int[][] matrix = new int[10][10];
                        int label = value._2.label;
                        int label_pred = value._1;
                        matrix[label_pred][label] = 1;
                        return matrix;
                    })
                    .reduce((m1, m2) -> {
                        int[][] matrix = new int[10][10];
                        for (int k = 0; k < 10; k++) {
                            for (int j = 0; j < 10; j++) {
                                matrix[k][j] = m1[k][j] + m2[k][j];
                            }
                        }
                        return matrix;
                    });

            double purity = RunKmeans.calculatePurity(finalResult);
            System.out.format("Final Purity = %.4f\n", purity);
        }
        else if (args[2].equals("communication")) {
            JavaRDD<RunKmeans.Point> pointRDD = sc.parallelize(pointList);
            JavaRDD<RunKmeans.Centroid> centroidRDD = sc.parallelize(centroidList);

            for (int i = 0; i < epoch; i++) {
                List<RunKmeans.Centroid> centroids = centroidRDD.collect();
                Broadcast<List<RunKmeans.Centroid>> centroidBroadcast = sc.broadcast(centroids);
                JavaRDD<Tuple2<Integer, RunKmeans.Point>> clusteredPointDataSet = pointRDD
                        // 寻找最近的聚类中心
                        .map((RunKmeans.Point p) -> {
                            double minDistance = Double.MAX_VALUE;
                            int closestCentroidId = -1;
                            // 检查所有的聚类中心
                            for (RunKmeans.Centroid centroid : centroidBroadcast.getValue()) {
                                // 计算每个点与聚类中心的距离（欧式距离）
                                double distance = p.euclideanDistance(centroid);

                                // 满足条件更新最近的聚类中心Id
                                if (distance < minDistance) {
                                    minDistance = distance;
                                    closestCentroidId = centroid.id;
                                }
                            }

                            // 生成一个包含聚类中心id和数据点的元组tuple.
                            return new Tuple2<>(closestCentroidId, p);
                        });

                int[][] result = clusteredPointDataSet
                        .map(value -> {
                            int[][] matrix = new int[10][10];
                            int label = value._2.label;
                            int label_pred = value._1;
                            matrix[label_pred][label] = 1;
                            return matrix;
                        })
                        .reduce((m1, m2) -> {
                            int[][] matrix = new int[10][10];
                            for (int k = 0; k < 10; k++) {
                                for (int j = 0; j < 10; j++) {
                                    matrix[k][j] = m1[k][j] + m2[k][j];
                                }
                            }
                            return matrix;
                        });

                double purity = RunKmeans.calculatePurity(result);
                System.out.format("Purity = %.4f\n", purity);


                centroidRDD = clusteredPointDataSet
                        .map(t -> new Tuple3<>(t._1, t._2, 1L))
                        .keyBy(t -> t._1())
                        // 求每一类点的坐标和
                        .reduceByKey((val1, val2) -> new Tuple3<>(val1._1(), val1._2().add(val2._2()), val1._3() + val2._3()))
                        // 求新的聚类中心
                        .map((value) -> new RunKmeans.Centroid(value._2._1(), value._2._2().div(value._2._3())));

            }

            List<RunKmeans.Centroid> centroids = centroidRDD.collect();
            Broadcast<List<RunKmeans.Centroid>> centroidBroadcast = sc.broadcast(centroids);
            int[][] finalResult = pointRDD
                    .map((RunKmeans.Point p) -> {
                        double minDistance = Double.MAX_VALUE;
                        int closestCentroidId = -1;

                        // 检查所有的聚类中心
                        for (RunKmeans.Centroid centroid : centroidBroadcast.getValue()) {
                            // 计算每个点与聚类中心的距离（欧式距离）
                            double distance = p.euclideanDistance(centroid);

                            // 满足条件更新最近的聚类中心Id
                            if (distance < minDistance) {
                                minDistance = distance;
                                closestCentroidId = centroid.id;
                            }
                        }

                        // 生成一个包含聚类中心id和数据点的元组tuple.
                        return new Tuple2<>(closestCentroidId, p);
                    })
                    .map(value -> {
                        int[][] matrix = new int[10][10];
                        int label = value._2.label;
                        int label_pred = value._1;
                        matrix[label_pred][label] = 1;
                        return matrix;
                    })
                    .reduce((m1, m2) -> {
                        int[][] matrix = new int[10][10];
                        for (int k = 0; k < 10; k++) {
                            for (int j = 0; j < 10; j++) {
                                matrix[k][j] = m1[k][j] + m2[k][j];
                            }
                        }
                        return matrix;
                    });

            double purity = RunKmeans.calculatePurity(finalResult);
            System.out.format("Final Purity = %.4f\n", purity);
        }
        else if (args[2].equals("no-broadcast")) {
            JavaRDD<RunKmeans.Point> pointRDD = sc.parallelize(pointList);
            JavaRDD<RunKmeans.Centroid> centroidRDD = sc.parallelize(centroidList);

            for (int i = 0; i < epoch; i++) {
                List<RunKmeans.Centroid> centroids = centroidRDD.collect();

                centroidRDD = pointRDD
                        // 寻找最近的聚类中心
                        .map((RunKmeans.Point p) -> {
                            double minDistance = Double.MAX_VALUE;
                            int closestCentroidId = -1;
                            // 检查所有的聚类中心
                            for (RunKmeans.Centroid centroid : centroids) {
                                // 计算每个点与聚类中心的距离（欧式距离）
                                double distance = p.euclideanDistance(centroid);

                                // 满足条件更新最近的聚类中心Id
                                if (distance < minDistance) {
                                    minDistance = distance;
                                    closestCentroidId = centroid.id;
                                }
                            }

                            // 生成一个包含聚类中心id和数据点的元组tuple.
                            return new Tuple2<>(closestCentroidId, p);
                        })
                        .map(t -> new Tuple3<>(t._1, t._2, 1L))
                        .keyBy(t -> t._1())
                        // 求每一类点的坐标和
                        .reduceByKey((val1, val2) -> new Tuple3<>(val1._1(), val1._2().add(val2._2()), val1._3() + val2._3()))
                        // 求新的聚类中心
                        .map((value) -> new RunKmeans.Centroid(value._2._1(), value._2._2().div(value._2._3())));
            }

            List<RunKmeans.Centroid> centroids = centroidRDD.collect();
            Broadcast<List<RunKmeans.Centroid>> centroidBroadcast = sc.broadcast(centroids);
            int[][] finalResult = pointRDD
                    .map((RunKmeans.Point p) -> {
                        double minDistance = Double.MAX_VALUE;
                        int closestCentroidId = -1;

                        // 检查所有的聚类中心
                        for (RunKmeans.Centroid centroid : centroidBroadcast.getValue()) {
                            // 计算每个点与聚类中心的距离（欧式距离）
                            double distance = p.euclideanDistance(centroid);

                            // 满足条件更新最近的聚类中心Id
                            if (distance < minDistance) {
                                minDistance = distance;
                                closestCentroidId = centroid.id;
                            }
                        }

                        // 生成一个包含聚类中心id和数据点的元组tuple.
                        return new Tuple2<>(closestCentroidId, p);
                    })
                    .map(value -> {
                        int[][] matrix = new int[10][10];
                        int label = value._2.label;
                        int label_pred = value._1;
                        matrix[label_pred][label] = 1;
                        return matrix;
                    })
                    .reduce((m1, m2) -> {
                        int[][] matrix = new int[10][10];
                        for (int k = 0; k < 10; k++) {
                            for (int j = 0; j < 10; j++) {
                                matrix[k][j] = m1[k][j] + m2[k][j];
                            }
                        }
                        return matrix;
                    });

            double purity = RunKmeans.calculatePurity(finalResult);
            System.out.format("Final Purity = %.4f\n", purity);
        }
        else {
            System.out.println("Unknown job type");
        }



    }
}
