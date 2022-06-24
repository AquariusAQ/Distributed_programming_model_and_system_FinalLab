package cn.edu.ecnu.finallab.model;

import java.io.Serializable;
import java.util.ArrayList;

class Matmul implements Serializable {
    private ArrayList<ArrayList<Double>> memX;
    private ArrayList<ArrayList<Double>> memW;
    Matmul() {
        memX = new ArrayList<>();
        memW = new ArrayList<>();
    }

    public ArrayList<ArrayList<Double>> forward(ArrayList<ArrayList<Double>> x, ArrayList<ArrayList<Double>> W) {
        memX = Utils.copyMatrix(x);
        memW = Utils.copyMatrix(W);
        ArrayList<ArrayList<Double>> h = Utils.matmul(x, W);
        return h;
    }

    public ArrayList<ArrayList<ArrayList<Double>>> backward(ArrayList<ArrayList<Double>> grad_y) {
        ArrayList<ArrayList<Double>> grad_x = Utils.matmul(grad_y, Utils.transpose(memW));
        ArrayList<ArrayList<Double>> grad_W = Utils.matmul(Utils.transpose(memX), grad_y);
        ArrayList<ArrayList<ArrayList<Double>>> result = new ArrayList<>();
        result.add(grad_x);
        result.add(grad_W);
//        System.out.println("!!! " + Utils.transpose(memX) + "\n" + grad_y);
        return result;
    }

//    public void setCache(ArrayList<ArrayList<Double>> x, ArrayList<ArrayList<Double>> w) {
//        memX = x;
//        memW = w;
//    }
//
//    public void getCache(ArrayList<ArrayList<Double>> x, ArrayList<ArrayList<Double>> w) {
//        x = memX;
//
//    }
}

class Relu implements Serializable {
    private ArrayList<ArrayList<Double>> memX;
    Relu() { memX = new ArrayList<>(); }

    public ArrayList<ArrayList<Double>> forward(ArrayList<ArrayList<Double>> x) {
        memX = Utils.copyMatrix(x);
        ArrayList<ArrayList<Double>> result = new ArrayList<>();
        for (int i = 0; i < x.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < x.get(0).size(); j++) {
                if (x.get(i).get(j) > 0) {
                    temp.add(x.get(i).get(j));
                } else {
                    temp.add(0.0);
                }
            }
            result.add(temp);
        }
        return result;
    }

    public ArrayList<ArrayList<Double>> backward(ArrayList<ArrayList<Double>> grad_y) {
        ArrayList<ArrayList<Double>> result = new ArrayList<>();
        for (int i = 0; i < grad_y.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < grad_y.get(0).size(); j++) {
                if (memX.get(i).get(j) > 0) {
                    temp.add(grad_y.get(i).get(j));
                } else {
                    temp.add(0.0);
                }
            }
            result.add(temp);
        }
        return result;
    }
}

class Softmax implements Serializable {
    final static private double EPSILON = 1e-8;
    private ArrayList<ArrayList<Double>> memOut;
    Softmax() { memOut = new ArrayList<>(); }

    public ArrayList<ArrayList<Double>> forward(ArrayList<ArrayList<Double>> x) {
        ArrayList<ArrayList<Double>> xExp = new ArrayList<>();
        for (int i = 0; i < x.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < x.get(0).size(); j++) {
                temp.add(Math.exp(x.get(i).get(j)));
            }
            xExp.add(temp);
        }

        ArrayList<Double> expSum = new ArrayList<>();
        for (int i = 0; i < x.size(); i++) {
            double temp = 0;
            for (int j = 0; j < x.get(0).size(); j++) {
                temp += xExp.get(i).get(j);
            }
            expSum.add(temp);
        }

        ArrayList<ArrayList<Double>> out = new ArrayList<>();
        for (int i = 0; i < x.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < x.get(0).size(); j++) {
                Double d = xExp.get(i).get(j) / (expSum.get(i) + EPSILON);
                temp.add(d);
            }
            out.add(temp);
        }
        memOut = out;
        return out;
    }

    public ArrayList<ArrayList<Double>> backward(ArrayList<ArrayList<Double>> grad_y) {
        ArrayList<ArrayList<Double>> grad0 = new ArrayList<>();
        for (int i = 0; i < grad_y.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < grad_y.get(0).size(); j++) {
                temp.add(grad_y.get(i).get(j) * memOut.get(i).get(j));
            }
            grad0.add(temp);
        }

        ArrayList<Double> gradSum = new ArrayList<>();
        for (int i = 0; i < grad_y.size(); i++) {
            double temp = 0;
            for (int j = 0; j < grad_y.get(0).size(); j++) {
                temp += grad0.get(i).get(j);
            }
            gradSum.add(temp);
        }

        ArrayList<ArrayList<Double>> result = new ArrayList<>();
        for (int i = 0; i < grad_y.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < grad_y.get(0).size(); j++) {
                temp.add(grad0.get(i).get(j) - memOut.get(i).get(j) * gradSum.get(i));
            }
            result.add(temp);
        }
        return result;
    }
}

class Log implements Serializable {
    final static private double EPSILON = 1e-12;
    private ArrayList<ArrayList<Double>> memX;
    Log() { memX = new ArrayList<>(); }

    public ArrayList<ArrayList<Double>> forward(ArrayList<ArrayList<Double>> x) {
        memX = Utils.copyMatrix(x);
        ArrayList<ArrayList<Double>> result = new ArrayList<>();
        for (int i = 0; i < x.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < x.get(0).size(); j++) {
                temp.add(Math.log(x.get(i).get(j)) + EPSILON);
            }
            result.add(temp);
        }
        return result;
    }

    public ArrayList<ArrayList<Double>> backward(ArrayList<ArrayList<Double>> grad_y) {
        ArrayList<ArrayList<Double>> result = new ArrayList<>();
        for (int i = 0; i < grad_y.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < grad_y.get(0).size(); j++) {
                temp.add(1 / (memX.get(i).get(j) + EPSILON) * grad_y.get(i).get(j));
            }
            result.add(temp);
        }
        return result;
    }
}

class BatchNorm implements Serializable {
    static final double EPS = 1e-5;
    static final double GAMMA = 1.;
    static final double BETA = 0.;
    private ArrayList<Double> gamma_s;
    private ArrayList<ArrayList<Double>> out;

    BatchNorm() { gamma_s = new ArrayList<>(); out = new ArrayList<>(); }

    public ArrayList<ArrayList<Double>> forward(ArrayList<ArrayList<Double>> x) {
        ArrayList<Double> meanList = new ArrayList<>();
        for (int i = 0; i < x.get(0).size(); i++) {
            Double sum = 0.;
            for (int j = 0; j < x.size(); j++) {
                sum += x.get(j).get(i);
            }
            meanList.add(sum / x.size());
        }

        ArrayList<Double> stdList = new ArrayList<>();
        for (int i = 0; i < x.get(0).size(); i++) {
            Double var = 0.;
            for (int j = 0; j < x.size(); j++) {
                var += (x.get(j).get(i) - meanList.get(i)) * (x.get(j).get(i) - meanList.get(i));
            }
            double std = Math.sqrt(var / x.size() + EPS);
            stdList.add(std);
            gamma_s.add(GAMMA / std);
        }

        ArrayList<ArrayList<Double>> y = new ArrayList<>();
        for (int i = 0; i < x.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < x.get(0).size(); j++) {
                temp.add((x.get(i).get(j) - meanList.get(j)) / stdList.get(j) * GAMMA + BETA);
            }
            y.add(temp);
        }
        out = y;
        return y;
    }

    public ArrayList<ArrayList<Double>> backward(ArrayList<ArrayList<Double>> grad) {
        ArrayList<Double> betaGrad = new ArrayList<>();
        for (int i = 0; i < grad.get(0).size(); i++) {
            Double sum = 0.;
            for (int j = 0; j < grad.size(); j++) {
                sum += grad.get(j).get(i);
            }
            betaGrad.add(sum / grad.size());
        }

        ArrayList<Double> gammaGrad = new ArrayList<>();
        for (int i = 0; i < grad.get(0).size(); i++) {
            Double sum = 0.;
            for (int j = 0; j < grad.size(); j++) {
                sum += grad.get(j).get(i) * out.get(j).get(i);
            }
            gammaGrad.add(sum / grad.size());
        }

//        self.gamma_s * (eta - self.y * gamma_grad - beta_grad)
        ArrayList<ArrayList<Double>> result = new ArrayList<>();
        for (int i = 0; i < grad.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < grad.get(0).size(); j++) {
                temp.add(gamma_s.get(j) * (grad.get(i).get(j) - out.get(i).get(j) * gammaGrad.get(j) - betaGrad.get(j)));
            }
            result.add(temp);
        }
        return result;
    }
}

public class MLP implements Serializable {
    private ArrayList<ArrayList<Double>> W1;
    private ArrayList<ArrayList<Double>> W2;
    private ArrayList<ArrayList<Double>> grad_W1;
    private ArrayList<ArrayList<Double>> grad_W2;
    private Matmul mul_h1;
    private Matmul mul_h2;
    private Relu relu;
    private Softmax softmax;
    private Log log;
    private BatchNorm bn1;
    private BatchNorm bn2;

    public ArrayList<ArrayList<Double>> h2_log;

    private double lr;
    private int inputs;
    private int outputs;
    private int hiddens;

    public MLP(int num_inputs, int num_outputs, int num_hiddens, double learning_rate) {
        // int num_hiddens = 100, double lr = 1e-5
        W1 = new ArrayList<>();
        W2 = new ArrayList<>();
        mul_h1 = new Matmul();
        mul_h2 = new Matmul();
        relu = new Relu();
        softmax = new Softmax();
        log = new Log();
        bn1 = new BatchNorm();
        bn2 = new BatchNorm();

        lr = learning_rate;
        inputs = num_inputs;
        outputs = num_outputs;
        hiddens = num_hiddens;

        for (int i = 0; i < num_inputs + 1; i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < num_hiddens; j++) {
                temp.add(Math.random());
            }
            W1.add(temp);
        }

        for (int i = 0; i < num_hiddens; i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < num_outputs; j++) {
                temp.add(Math.random());
            }
            W2.add(temp);
        }
    }

    public ArrayList<ArrayList<Double>> forward(ArrayList<ArrayList<Double>> x) {
        ArrayList<ArrayList<Double>> x_0 = new ArrayList<>();
        for (int i = 0; i < x.size(); i++) {
            ArrayList<Double> temp = new ArrayList<>();
            temp.addAll(x.get(i));
            temp.add(1.0);
            x_0.add(temp);
        }

        ArrayList<ArrayList<Double>> x_1 = mul_h1.forward(x_0, W1);
        ArrayList<ArrayList<Double>> x_bn_1 = bn1.forward(x_1);

        ArrayList<ArrayList<Double>> x_2 = relu.forward(x_bn_1);

        ArrayList<ArrayList<Double>> x_3 = mul_h2.forward(x_2, W2);
        ArrayList<ArrayList<Double>> x_bn_2 = bn2.forward(x_3);

        ArrayList<ArrayList<Double>> x_4 = softmax.forward(x_bn_2);

        h2_log = log.forward(x_4);
        return h2_log;
    }

    public void backward(ArrayList<ArrayList<Double>> label ){

        ArrayList<ArrayList<Double>> y = Utils.copyMatrix(label);

        ArrayList<ArrayList<Double>> y_0 = log.backward(y);

        ArrayList<ArrayList<Double>> y_1 = softmax.backward(y_0);

        ArrayList<ArrayList<Double>> y_bn_2 = bn2.backward(y_1);

        ArrayList<ArrayList<ArrayList<Double>>> dataA = mul_h2.backward(y_bn_2);
        ArrayList<ArrayList<Double>> grad_x2 = dataA.get(0);
        grad_W2 = dataA.get(1);

        ArrayList<ArrayList<Double>> y_2 = relu.backward(grad_x2);

        ArrayList<ArrayList<Double>> y_bn_1 = bn1.backward(y_2);

        ArrayList<ArrayList<ArrayList<Double>>> dataB = mul_h1.backward(y_bn_1);
        ArrayList<ArrayList<Double>> grad_x1 = dataB.get(0);
        grad_W1 = dataB.get(1);
    }

    public void update() {
        ArrayList<ArrayList<Double>> W1New = new ArrayList<>();
        for (int i = 0; i < inputs + 1; i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < hiddens; j++) {
                temp.add(W1.get(i).get(j) + lr * grad_W1.get(i).get(j));
            }
            W1New.add(temp);
        }
        W1 = W1New;

        ArrayList<ArrayList<Double>> W2New = new ArrayList<>();
        for (int i = 0; i < hiddens; i++) {
            ArrayList<Double> temp = new ArrayList<>();
            for (int j = 0; j < outputs; j++) {
                temp.add(W2.get(i).get(j) + lr * grad_W2.get(i).get(j));
            }
            W2New.add(temp);
        }
        W2 = W2New;
    }
}
