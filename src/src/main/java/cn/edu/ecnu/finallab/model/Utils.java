package cn.edu.ecnu.finallab.model;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

import java.util.ArrayList;

public class Utils {
    public static Matrix PCA(Matrix X, int component) {
        int rows = X.getRowDimension();
        int columns = X.getColumnDimension();

        Matrix Z = X.transpose();
        Z = Z.times(X);
        Matrix eigenVectorMatrix = Z.eig().getV();

        int[] index = new int[component];
        for (int i = 0; i < component; i++) {
            index[i] = columns - 1 - i;
        }

        eigenVectorMatrix = eigenVectorMatrix.getMatrix(0, rows-1, index);

        Matrix newX = X.times(eigenVectorMatrix).times(eigenVectorMatrix.transpose());

        return newX;
    }

    public static double RMSE(Matrix m1, Matrix m2) {
        if (m1.getRowDimension() != m2.getRowDimension() || m1.getColumnDimension() != m2.getColumnDimension()) {
            System.out.println("Matric size is not same!");
        }

        double sum = 0.;
        double[][] d1 = m1.getArray();
        double[][] d2 = m2.getArray();
        for (int i = 0; i < m1.getRowDimension(); i++) {
            for (int j = 0; j < m1.getColumnDimension(); j++) {
                sum += (d1[i][j] - d2[i][j]) * (d1[i][j] - d2[i][j]);
            }
        }
        return Math.sqrt(sum / (m1.getRowDimension() * m1.getColumnDimension()));
    }

    public static double[][] standardization(double[][] m) {
        int shape0 = m.length;
        int shape1 = m[0].length;

        double[] mean = new double[shape0];

        for (int i = 0; i < shape0; i++) {
            double sum = 0.;
            for (int j = 0; j < shape1; j++) {
                sum += m[i][j];
            }
            mean[i] = sum / shape1;
        }

        double[] std = new double[shape0];
        for (int i = 0; i < shape0; i++) {
            double var = 0.;
            for (int j = 0; j < shape1; j++) {
                var += (m[i][j] - mean[i]) * (m[i][j] - mean[i]);
            }
            std[i] = Math.sqrt(var / shape1);
        }

        double[][] result = new double[shape0][shape1];
        for (int i = 0; i < shape0; i++) {
            for (int j = 0; j < shape1; j++) {
                if (std[i] == 0) {
                    result[i][j] = m[i][j] - mean[i];
                } else {
                    result[i][j] = (m[i][j] - mean[i]) / std[i];
                }
            }
        }

        return result;
    }

    public static void printMatrix(Matrix m) {
        double[][] arr = m.getArray();
        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                System.out.print("" + arr[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static ArrayList<ArrayList<Double>> copyMatrix(ArrayList<ArrayList<Double>> x) {
        ArrayList<ArrayList<Double>> result = new ArrayList<>();
        for (int i = 0; i < x.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>();
            temp.addAll(x.get(i));
            result.add(temp);
        }
        return result;
    }

    public static ArrayList<ArrayList<Double>> matmul(ArrayList<ArrayList<Double>> x, ArrayList<ArrayList<Double>> y) {
//        System.out.format("%d * %d , %d * %d\n", x.size(), x.get(0).size(), y.size(), y.get(0).size());
        if (x.get(0).size() != y.size()) {
            System.out.println("Matrix size " + x.get(0).size() + "!=" + y.size());
            System.exit(-2);
        }
        int a = x.size();
        int b = x.get(0).size();
        int c = y.get(0).size();
        ArrayList<ArrayList<Double>> z = new ArrayList<>();
        for (int i = 0; i < a; i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < c; j++) {
                double value = 0;
                for (int k = 0; k < b; k++) {
                    value += x.get(i).get(k) * y.get(k).get(j);
                }
                temp.add(value);
            }
            z.add(temp);
        }
        return z;
    }

    public static ArrayList<ArrayList<Double>> transpose(ArrayList<ArrayList<Double>> x) {
        int a = x.size();
        int b = x.get(0).size();
        ArrayList<ArrayList<Double>> result = new ArrayList<>();
        for (int j = 0; j < b; j++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int i = 0; i < a; i++) {
                temp.add(x.get(i).get(j));
            }
            result.add(temp);
        }
        return result;
    }

    public static double computeLoss(ArrayList<ArrayList<Double>> log_prob, ArrayList<ArrayList<Double>> labels) {
        int len = log_prob.size();
        double loss = 0;
        for (int i = 0; i < log_prob.size(); i++) {
            for (int j = 0; j < log_prob.get(0).size(); j++) {
//                if (log_prob.get(i).get(j).isInfinite()) {
//                    continue;
//                }
                Double l = log_prob.get(i).get(j) * labels.get(i).get(j);
                if (!l.isNaN()) {
                    loss -= l;
                }
            }
        }
        return loss / len;
    }

    public static int argmax(ArrayList<Double> a) {
        double max = Double.NEGATIVE_INFINITY;
        int index = -1;
        for (int i = 0; i < a.size(); i++) {
            if (a.get(i) > max) {
                max = a.get(i);
                index = i;
            }
        }
        return index;
    }

    public static double computeAccuracy(ArrayList<ArrayList<Double>> log_prob, ArrayList<ArrayList<Double>> labels) {
        int len = log_prob.size();
        int sum = 0;
        for (int i = 0; i < len; i++) {
            if (argmax(log_prob.get(i)) == argmax(labels.get(i))) {
                sum += 1;
            }
        }
        return sum * 1.0 / len;
    }
}
