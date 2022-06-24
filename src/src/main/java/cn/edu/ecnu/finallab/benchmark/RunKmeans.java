package cn.edu.ecnu.finallab.benchmark;

import cn.edu.ecnu.finallab.MnistData;
import cn.edu.ecnu.finallab.benchmark.flink.KmeansFlink;
import cn.edu.ecnu.finallab.benchmark.spark.KmeansSpark;

import java.io.Serializable;
import java.util.ArrayList;

public class RunKmeans {

    public static double run(String[] args) throws Exception {
        final String imagePath = args[4];
        final String labelPath = args[5];

        int K = 10;
        int epoch = Integer.parseInt(args[3]);

        System.out.println("Running K-means on MNIST.");
        System.out.format("K: %d, Num Epoch: %d\n", K, epoch);

        MnistData data = new MnistData(imagePath, labelPath);
        int numSamples = data.getNumSamples();
        ArrayList<Integer> centroidId = new ArrayList<>();
        int i = 0;
        while (i < K) {
            int id = (int)Math.floor(Math.random() * numSamples);
            if (!centroidId.contains(id)) {
                centroidId.add(id);
                i += 1;
            }
        }

        ArrayList<Point> pointList = new ArrayList<>();
        ArrayList<Centroid> centroidsList = new ArrayList<>();
        int k = 0;
        for (int j = 0; j < numSamples; j++) {
            if (centroidId.contains(j)) {
                centroidsList.add(new Centroid(k, data.getImageByIndex(j), data.getLiteralLabelByIndex(j)));
                k += 1;
            }
            pointList.add(new Point(data.getImageByIndex(j), data.getLiteralLabelByIndex(j)));
        }

        double usedTime = 0.;

        if (args[1].equals("spark")) {
            System.out.println("Training with Spark...");

            long startTime = System.currentTimeMillis();

            KmeansSpark.runSpark(args, centroidsList, pointList, epoch);

            long endTime = System.currentTimeMillis();
            usedTime = (endTime - startTime) * 1.0 / 1000;
        }
        else if (args[1].equals("flink")) {
            System.out.println("Training with Flink (without Bulk Iteration)...");

            long startTime = System.currentTimeMillis();

            KmeansFlink.runFlink(args, centroidsList, pointList, epoch);

            long endTime = System.currentTimeMillis();
            usedTime = (endTime - startTime) * 1.0 / 1000;
        }
        else if (args[1].equals("flink-bulk")) {
            System.out.println("Training with Flink (with Bulk Iteration)...");

            long startTime = System.currentTimeMillis();

            KmeansFlink.runFlinkBulk(args, centroidsList, pointList, epoch);

            long endTime = System.currentTimeMillis();
            usedTime = (endTime - startTime) * 1.0 / 1000;
        }

        return usedTime;
    }

    public static double calculatePurity(int[][] matrix) {
        int count = matrix.length;

//        for (int i = 0; i < count; i++) {
//            for (int j = 0; j < count; j++) {
//                System.out.format("%d\t", matrix[i][j]);
//            }
//            System.out.println();
//        }

        int[] ni_ = new int[count];
        int[] n_j = new int[count];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < count; j++) {
                ni_[i] += matrix[i][j];
                n_j[j] += matrix[i][j];
            }
        }
        int n = 0;
        for (int i = 0; i < count; i++) {
            n += ni_[i];
        }
        double E = 0.;
        double P = 0.;
        for (int i = 0; i < count; i++) {
            double temp = 0;
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < count; j++) {
                if (matrix[i][j] != 0) {
                    if (matrix[i][j] > 0 && ni_[i] > 0) {
                        temp += matrix[i][j] * 1.0 / ni_[i] * Math.log(matrix[i][j] * 1.0 / ni_[i]);
                    }
                }
                if (matrix[i][j] * 1.0 / ni_[i] > max) {
                    max = matrix[i][j] * 1.0 / ni_[i];
                }
            }
            E += ni_[i] * 1.0 / n * temp;
            P += ni_[i] * 1.0 / n * max;
        }
        E = -E;
        return P;
    }

//    public static void main(String[] args) {
//        int[][] test = {{100,100,1},{10,1100,100},{100,1,110}};
//        System.out.println(calculatePurity(test));
//    }

    public static class Point implements Serializable {

        public ArrayList<Double> X;
        public int label;

        public Point() {}

        public Point(ArrayList<Double> X0, int label) {
            X = new ArrayList<>();
            this.X.addAll(X0);
            this.label = label;
        }

        public Point add(Point other) {
            ArrayList<Double> newX = new ArrayList<>();
            for (int i = 0; i < X.size(); i++) {
                newX.add(X.get(i) + other.X.get(i));
            }
            X = newX;
            return this;
        }

        public Point div(long val) {
            ArrayList<Double> newX = new ArrayList<>();
            for (int i = 0; i < X.size(); i++) {
                newX.add(X.get(i) / val);
            }
            X = newX;
            return this;
        }

        public double euclideanDistance(Point other) {
            Double distance = 0.;
            for (int i = 0; i < X.size(); i++) {
                distance += (X.get(i) - other.X.get(i)) * (X.get(i) - other.X.get(i));
            }
            return Math.sqrt(distance);
        }


        public void clear() {
            for (int i = 0; i < X.size(); i++) {
                X.set(i, 0.);
            }
        }

        @Override
        public String toString() {
            return " label: " + label + ", X: " + X.toString();
        }
    }

    public static class Centroid extends Point {

        public int id;

        public Centroid() {}

        public Centroid(int id, ArrayList<Double> X, int label) {
            super(X, label);
            this.id = id;
        }

        public Centroid(int id, Point p) {
            super(p.X, p.label);
            this.id = id;
        }

        @Override
        public String toString() {
            return id + " " + super.toString();
        }
    }

}
