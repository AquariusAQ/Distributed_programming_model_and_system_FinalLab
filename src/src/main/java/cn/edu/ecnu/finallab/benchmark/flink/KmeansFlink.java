package cn.edu.ecnu.finallab.benchmark.flink;

import cn.edu.ecnu.finallab.Run;
import cn.edu.ecnu.finallab.benchmark.RunKmeans;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.functions.FunctionAnnotation;
import org.apache.flink.api.java.operators.IterativeDataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;

import java.util.ArrayList;
import java.util.List;

public class KmeansFlink {
    public static void runFlink(String[] args, ArrayList<RunKmeans.Centroid> centroidsList,
                                  ArrayList<RunKmeans.Point> pointList, int epoch) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        if (args[2].equals("vanilla")) {
            DataSet<RunKmeans.Point> pointDataSet = env.fromCollection(pointList);
            DataSet<RunKmeans.Centroid> centroidDataSet = env.fromCollection(centroidsList);


            for (int i = 0; i < epoch; i++) {

                centroidDataSet = pointDataSet
                        // 寻找最近的聚类中心
                        .map(new SelectNearestCenter()).withBroadcastSet(centroidDataSet, "centroids")
                        .map(new CountAppender())
                        .groupBy(0)
                        // 求每一类点的坐标和
                        .reduce(new CentroidAccumulator())
                        // 求新的聚类中心
                        .map(new CentroidAverager());

            }

            DataSet<int[][]> finalResult = pointDataSet
                    .map(new SelectNearestCenter()).withBroadcastSet(centroidDataSet, "centroids")
                    .map(new ContingencyMatrix())
                    .reduce(new ContingencyMatrixSum());

            double purity = RunKmeans.calculatePurity(finalResult.collect().get(0));
            System.out.format("Final Purity = %.4f\n", purity);
        }
        else if (args[2].equals("communication")) {
            DataSet<RunKmeans.Point> pointDataSet = env.fromCollection(pointList);
            DataSet<RunKmeans.Centroid> centroidDataSet = env.fromCollection(centroidsList);

//        IterativeDataSet<RunKmeans.Centroid> iterate = centroidDataSet.iterate(epoch);

            for (int i = 0; i < epoch; i++) {
                DataSet<Tuple2<Integer, RunKmeans.Point>> clusteredPointDataSet = pointDataSet
                        // 寻找最近的聚类中心
                        .map(new SelectNearestCenter()).withBroadcastSet(centroidDataSet, "centroids");

                DataSet<int[][]> result = clusteredPointDataSet
                        .map(new ContingencyMatrix())
                        .reduce(new ContingencyMatrixSum())
                        .map((matrix) -> {
                            double purity = RunKmeans.calculatePurity(matrix);
                            System.out.format("Purity = %.4f\n", purity);
                            return matrix;
                        });

                centroidDataSet = clusteredPointDataSet
                        .map(new CountAppender())
                        .groupBy(0)
                        // 求每一类点的坐标和
                        .reduce(new CentroidAccumulator())
                        // 求新的聚类中心
                        .map(new CentroidAverager());

                if (result.count() == 0) {
                    break;
                }
            }

            DataSet<int[][]> finalResult = pointDataSet
                    .map(new SelectNearestCenter()).withBroadcastSet(centroidDataSet, "centroids")
                    .map(new ContingencyMatrix())
                    .reduce(new ContingencyMatrixSum());

            double purity = RunKmeans.calculatePurity(finalResult.collect().get(0));
            System.out.format("Final Purity = %.4f\n", purity);
        }
        else if (args[2].equals("no-broadcast")) {
            DataSet<RunKmeans.Point> pointDataSet = env.fromCollection(pointList);
            DataSet<RunKmeans.Centroid> centroidDataSet = env.fromCollection(centroidsList);


            for (int i = 0; i < epoch; i++) {

                List<RunKmeans.Centroid> centroids = centroidDataSet.collect();
                centroidDataSet = env.fromCollection(centroids);

                centroidDataSet = pointDataSet
                        // 寻找最近的聚类中心
                        .map(new SelectNearestCenter()).withBroadcastSet(centroidDataSet, "centroids")
                        .map(new CountAppender())
                        .groupBy(0)
                        // 求每一类点的坐标和
                        .reduce(new CentroidAccumulator())
                        // 求新的聚类中心
                        .map(new CentroidAverager());

            }

            DataSet<int[][]> finalResult = pointDataSet
                    .map(new SelectNearestCenter()).withBroadcastSet(centroidDataSet, "centroids")
                    .map(new ContingencyMatrix())
                    .reduce(new ContingencyMatrixSum());

            double purity = RunKmeans.calculatePurity(finalResult.collect().get(0));
            System.out.format("Final Purity = %.4f\n", purity);
        }
        else {
            System.out.println("Unknown job type");
        }


    }

    public static void runFlinkBulk(String[] args, ArrayList<RunKmeans.Centroid> centroidsList,
                                ArrayList<RunKmeans.Point> pointList, int epoch) throws Exception {

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        if (args[2].equals("vanilla")) {
            DataSet<RunKmeans.Point> pointDataSet = env.fromCollection(pointList);
            DataSet<RunKmeans.Centroid> centroidDataSet = env.fromCollection(centroidsList);

            IterativeDataSet<RunKmeans.Centroid> iterate = centroidDataSet.iterate(epoch);

            DataSet<RunKmeans.Centroid> newCentroid = pointDataSet
                    // 寻找最近的聚类中心
                    .map(new SelectNearestCenter()).withBroadcastSet(iterate, "centroids")
                    // 求每一类点的坐标和
                    .map(new CountAppender())
                    .groupBy(0)
                    .reduce(new CentroidAccumulator())
                    // 求新的聚类中心
                    .map(new CentroidAverager());

//        DataSet<>

            DataSet<RunKmeans.Centroid> finalCentroidDataSet = iterate.closeWith(newCentroid);

            DataSet<int[][]> finalResult = pointDataSet
                    .map(new SelectNearestCenter()).withBroadcastSet(finalCentroidDataSet, "centroids")
                    .map(new ContingencyMatrix())
                    .reduce(new ContingencyMatrixSum());

            double purity = RunKmeans.calculatePurity(finalResult.collect().get(0));
            System.out.format("Final Purity = %.4f\n", purity);
        }
        else if (args[2].equals("communication")) {
            DataSet<RunKmeans.Point> pointDataSet = env.fromCollection(pointList);
            DataSet<RunKmeans.Centroid> centroidDataSet = env.fromCollection(centroidsList);

            IterativeDataSet<RunKmeans.Centroid> iterate = centroidDataSet.iterate(epoch);

            DataSet<Tuple2<Integer, RunKmeans.Point>> clusteredPointDataSet = pointDataSet
                    // 寻找最近的聚类中心
                    .map(new SelectNearestCenter()).withBroadcastSet(iterate, "centroids");

            DataSet<int[][]> result = clusteredPointDataSet
                    .map(new ContingencyMatrix())
                    .reduce(new ContingencyMatrixSum())
                    .map((matrix) -> {
                        double purity = RunKmeans.calculatePurity(matrix);
                        System.out.format("Purity = %.4f\n", purity);
                        return matrix;
                    });

            DataSet<RunKmeans.Centroid> newCentroid = clusteredPointDataSet
                    // 求每一类点的坐标和
                    .map(new CountAppender())
                    .groupBy(0)
                    .reduce(new CentroidAccumulator())
                    // 求新的聚类中心
                    .map(new CentroidAverager());

//        DataSet<>

            DataSet<RunKmeans.Centroid> finalCentroidDataSet = iterate.closeWith(newCentroid, result);

            DataSet<int[][]> finalResult = pointDataSet
                    .map(new SelectNearestCenter()).withBroadcastSet(finalCentroidDataSet, "centroids")
                    .map(new ContingencyMatrix())
                    .reduce(new ContingencyMatrixSum());

            double purity = RunKmeans.calculatePurity(finalResult.collect().get(0));
            System.out.format("Final Purity = %.4f\n", purity);
        }
        else if (args[2].equals("no-broadcast")) {
            DataSet<RunKmeans.Point> pointDataSet = env.fromCollection(pointList);
            DataSet<RunKmeans.Centroid> centroidDataSet = env.fromCollection(centroidsList);

            IterativeDataSet<RunKmeans.Centroid> iterate = centroidDataSet.iterate(epoch);


            DataSet<RunKmeans.Centroid> newCentroid = pointDataSet
                    // 寻找最近的聚类中心
                    .map(new SelectNearestCenter()).withBroadcastSet(iterate, "centroids")
                    // 求每一类点的坐标和
                    .map(new CountAppender())
                    .groupBy(0)
                    .reduce(new CentroidAccumulator())
                    // 求新的聚类中心
                    .map(new CentroidAverager());

            DataSet<RunKmeans.Centroid> feedback = newCentroid
                    .reduce(new ReduceFunction<RunKmeans.Centroid>() {
                        @Override
                        public RunKmeans.Centroid reduce(RunKmeans.Centroid centroid, RunKmeans.Centroid t1) throws Exception {
                            return centroid;
                        }
                    });

//        DataSet<>

            DataSet<RunKmeans.Centroid> finalCentroidDataSet = iterate.closeWith(newCentroid, feedback);

            DataSet<int[][]> finalResult = pointDataSet
                    .map(new SelectNearestCenter()).withBroadcastSet(finalCentroidDataSet, "centroids")
                    .map(new ContingencyMatrix())
                    .reduce(new ContingencyMatrixSum());

            double purity = RunKmeans.calculatePurity(finalResult.collect().get(0));
            System.out.format("Final Purity = %.4f\n", purity);
        }
        else {
            System.out.println("Unknown job type");
        }


    }


    /** 从数据点确定最近的聚类中心. */
    @FunctionAnnotation.ForwardedFields("*->1")
    public static final class SelectNearestCenter extends RichMapFunction<RunKmeans.Point, Tuple2<Integer, RunKmeans.Point>> {
        private List<RunKmeans.Centroid> centroids;

        /** 从广播变量中读取聚类中心值到集合中. */
        @Override
        public void open(Configuration parameters) throws Exception {
            this.centroids = getRuntimeContext().getBroadcastVariable("centroids");
        }

        @Override
        public Tuple2<Integer, RunKmeans.Point> map(RunKmeans.Point p) throws Exception {

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
        }
    }

    /** 向tupel2追加计数变量. */
    @FunctionAnnotation.ForwardedFields("f0;f1")
    public static final class CountAppender implements MapFunction<Tuple2<Integer, RunKmeans.Point>, Tuple3<Integer, RunKmeans.Point, Long>> {

        @Override
        public Tuple3<Integer/*id*/, RunKmeans.Point, Long/*1L*/> map(Tuple2<Integer, RunKmeans.Point> t) {
            return new Tuple3<>(t.f0, t.f1, 1L);
        }
    }

    /** 求同一个类所有点的x,y坐标总数和计数点坐标. */
    //@FunctionAnnotation.ForwardedFields("0")
    public static final class CentroidAccumulator implements ReduceFunction<Tuple3<Integer, RunKmeans.Point, Long>> {

        @Override
        public Tuple3<Integer, RunKmeans.Point, Long> reduce(Tuple3<Integer, RunKmeans.Point, Long> val1, Tuple3<Integer, RunKmeans.Point, Long> val2) {
            return new Tuple3<>(val1.f0, val1.f1.add(val2.f1), val1.f2 + val2.f2);
        }
    }

    /** 从坐标和点的个数计算新的聚类中心. */
    //@FunctionAnnotation.ForwardedFields("0->id")
    public static final class CentroidAverager implements MapFunction<Tuple3<Integer/*id*/, RunKmeans.Point/*累加的坐标点*/, Long/*个数*/>, RunKmeans.Centroid> {

        @Override
        public RunKmeans.Centroid map(Tuple3<Integer, RunKmeans.Point, Long> value) {
            return new RunKmeans.Centroid(value.f0, value.f1.div(value.f2));
        }
    }

    /** 求可能性矩阵 **/
    public static final class ContingencyMatrix implements MapFunction<Tuple2<Integer, RunKmeans.Point>/*聚类点*/, int[][]/*可能性矩阵*/> {

        @Override
        public int[][] map(Tuple2<Integer, RunKmeans.Point> value) throws Exception {
            int[][] matrix = new int[10][10];
            int label = value.f1.label;
            int label_pred = value.f0;
            matrix[label_pred][label] = 1;
            return matrix;
        }
    }

    public static final class ContingencyMatrixSum implements ReduceFunction<int[][]> {

        @Override
        public int[][] reduce(int[][] m1, int[][] m2) throws Exception {
            int[][] matrix = new int[10][10];
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    matrix[i][j] = m1[i][j] + m2[i][j];
                }
            }
            return matrix;
        }
    }

}
