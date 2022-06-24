import cn.edu.ecnu.finallab.MnistData;
import cn.edu.ecnu.finallab.model.MLP;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;

public class TestMlp {
    static final String imagePath = "./input/mnist-images.csv";
    static final String labelPath = "./input/mnist-labels.csv";

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

    public static void main(String[] args) {
        int numHiddens = 20;
        double learningRate = 2;
        int epoch = 10;
        int batchSize = 64;

        MnistData data = new MnistData(imagePath, labelPath);
        ArrayList<ArrayList<ArrayList<Double>>> imageLoader = new ArrayList<>();
        ArrayList<ArrayList<ArrayList<Double>>> labelLoader = new ArrayList<>();
        data.getAllBatch(batchSize, imageLoader, labelLoader);
        int batchNum = data.getBatchNum();

        MLP mlp = new MLP(data.getNumFeatures(), 10, numHiddens, learningRate);
        for (int i = 0; i < epoch; i++) {
//            System.out.format("Data size: %d * %d\n", X.size(), X.get(0).size());
            double loss = 0.;
            double accuracy = 0.;
            for (int j = 0; j < batchNum; j++) {
                ArrayList<ArrayList<Double>> pred_y = mlp.forward(imageLoader.get(j));
                mlp.backward(labelLoader.get(j));
                mlp.update();
                loss += computeLoss(pred_y, labelLoader.get(j));
                accuracy += computeAccuracy(pred_y, labelLoader.get(j));
            }

            System.out.format("Epoch: %d, Loss: %.4f, Acc: %.4f\n", i+1, loss / batchNum, accuracy / batchNum);
        }

    }
}
