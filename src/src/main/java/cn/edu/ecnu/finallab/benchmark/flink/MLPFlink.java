package cn.edu.ecnu.finallab.benchmark.flink;

import cn.edu.ecnu.finallab.MnistData;
import cn.edu.ecnu.finallab.model.MLP;
import cn.edu.ecnu.finallab.model.Utils;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.DataSource;
import org.apache.flink.api.java.operators.IterativeDataSet;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

public class MLPFlink {
    public static void runFlink(String[] args, MnistData data, ArrayList<ArrayList<ArrayList<Double>>> imageLoader,
                                ArrayList<ArrayList<ArrayList<Double>>> labelLoader,
                                int numHiddens, double learningRate, int batchNum, int epoch) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        MLP mlp = new MLP(data.getNumFeatures(), 10, numHiddens, learningRate);

        ArrayList<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>> dataTuple = new ArrayList<>();
        for (int i = 0; i < batchNum; i++) {
            ArrayList<ArrayList<ArrayList<Double>>> sampleData = new ArrayList<>();
            sampleData.add(imageLoader.get(i));
            sampleData.add(labelLoader.get(i));
            Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> x =
                    new Tuple2<>(mlp, sampleData);
            dataTuple.add(x);
        }

        DataSource<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>> dataSource = env.fromCollection(dataTuple);

        DataSet< Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> > dataset = dataSource.map(
                new MapFunction<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>, Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>>() {
                    @Override
                    public Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> map(Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> batch) throws Exception {
                        ArrayList<ArrayList<Double>> pred = new ArrayList<>();
                        ArrayList<ArrayList<ArrayList<Double>>> sampleData = batch._2;
                        sampleData.add(pred);
                        Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> result = new Tuple2<>(mlp, sampleData);
                        return result;
                    }
                }
        );


        for (int i = 0; i < epoch; i++) {
            dataset = dataset.map(
                    new MapFunction<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>, Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>>() {
                        @Override
                        public Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> map(Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> batch) throws Exception {
                            ArrayList<ArrayList<Double>> pred_y = batch._1.forward(batch._2.get(0));
                            batch._1.backward(batch._2.get(1));
                            batch._1.update();
                            ArrayList<ArrayList<ArrayList<Double>>> sampleData = batch._2;
                            sampleData.set(2, pred_y);
                            return new Tuple2<>(batch._1(), sampleData);
                        }
                    }
            );

            if (args[2].equals("communication")) {
                List<Tuple2<Double, Double>> resultDataset = dataset.map(
                        new MapFunction<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>, Tuple2<Double, Double>>() {
                            @Override
                            public Tuple2<Double, Double> map(Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> batch) throws Exception {
                                ArrayList<ArrayList<Double>> pred_y = batch._2.get(2);
                                double loss = Utils.computeLoss(pred_y, batch._2.get(1));
                                double accuracy = Utils.computeAccuracy(pred_y, batch._2.get(1));
                                return new Tuple2<>(loss, accuracy);
                            }
                        }
                )
                        .reduce((r1, r2) -> new Tuple2<>(r1._1+r2._1, r1._2+r2._2))
                        .collect();
                Tuple2<Double, Double> result = resultDataset.get(0);

                double loss = result._1 / batchNum;
                double accuracy = result._2 / batchNum;
                System.out.format("\tEpoch: %d, Loss: %.4f, Acc: %.4f\n", i+1, loss, accuracy);
            }

        }

        if (args[2].equals("vanilla")) {
            List<Tuple2<Double, Double>> resultDataset = dataset.map(
                    new MapFunction<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>, Tuple2<Double, Double>>() {
                        @Override
                        public Tuple2<Double, Double> map(Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> batch) throws Exception {
                            ArrayList<ArrayList<Double>> pred_y = batch._2.get(2);
                            double loss = Utils.computeLoss(pred_y, batch._2.get(1));
                            double accuracy = Utils.computeAccuracy(pred_y, batch._2.get(1));
                            return new Tuple2<>(loss, accuracy);
                        }
                    }
            )
                    .reduce((r1, r2) -> new Tuple2<>(r1._1+r2._1, r1._2+r2._2))
                    .collect();
            Tuple2<Double, Double> result = resultDataset.get(0);

            double loss = result._1 / batchNum;
            double accuracy = result._2 / batchNum;
            System.out.format("\tFinal Loss: %.4f, Acc: %.4f\n", loss, accuracy);
        }
    }

    public static void runFlinkBulk(String[] args, MnistData data, ArrayList<ArrayList<ArrayList<Double>>> imageLoader,
                                ArrayList<ArrayList<ArrayList<Double>>> labelLoader,
                                int numHiddens, double learningRate, int batchNum, int epoch) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        MLP mlp = new MLP(data.getNumFeatures(), 10, numHiddens, learningRate);

        ArrayList<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>> dataTuple = new ArrayList<>();
        for (int i = 0; i < batchNum; i++) {
            ArrayList<ArrayList<ArrayList<Double>>> sampleData = new ArrayList<>();
            sampleData.add(imageLoader.get(i));
            sampleData.add(labelLoader.get(i));
            Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> x =
                    new Tuple2<>(mlp, sampleData);
            dataTuple.add(x);
        }

        DataSource<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>> dataSource = env.fromCollection(dataTuple);

        DataSet< Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> > dataset = dataSource.map(
                new MapFunction<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>, Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>>() {
                    @Override
                    public Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> map(Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> batch) throws Exception {
                        ArrayList<ArrayList<Double>> pred = new ArrayList<>();
                        ArrayList<ArrayList<ArrayList<Double>>> sampleData = batch._2;
                        sampleData.add(pred);
                        Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> result = new Tuple2<>(mlp, sampleData);
                        return result;
                    }
                }
        );

        // 可迭代数据集
        IterativeDataSet<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>> iteration = dataset.iterate(epoch);

        // 每轮迭代后的回调函数
//            DataSet<Tuple2<Double, Double>> feedback = iteration.map(
//                    new MapFunction<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>, Tuple2<Double, Double>>() {
//                        @Override
//                        public Tuple2<Double, Double> map(Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> batch) throws Exception {
//                            ArrayList<ArrayList<Double>> pred_y = batch._2.get(2);
//                            double loss = Utils.computeLoss(pred_y, batch._2.get(1));
//                            double accuracy = Utils.computeAccuracy(pred_y, batch._2.get(1));
//                            return new Tuple2<>(loss, accuracy);
//                        }
//                    }
//            )
//                    .reduce((r1, r2) -> new Tuple2<>(r1._1+r2._1, r1._2+r2._2))
//                    .map((result) -> {
//                        double loss = result._1 / batchNum;
//                        double accuracy = result._2 / batchNum;
//                        System.out.format("\tLoss: %.4f, Acc: %.4f\n", loss, accuracy);
//                        return result;
//                    });

        // 迭代主体
        DataSet<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>> iterationBody = iteration.map(
                new RichMapFunction<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>, Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>>() {
                    @Override
                    public Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> map(Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>> batch) throws Exception {
                        ArrayList<ArrayList<Double>> pred_y = batch._1.forward(batch._2.get(0));
                        System.out.println(batch._1);
                        batch._1.backward(batch._2.get(1));
                        batch._1.update();
                        ArrayList<ArrayList<ArrayList<Double>>> sampleData = batch._2;
                        sampleData.set(2, pred_y);
                        return new Tuple2<>(batch._1(), sampleData);
                    }
                }
        );




        DataSet<Tuple2<MLP, ArrayList<ArrayList<ArrayList<Double>>>>> finalDataset = iteration.closeWith(iterationBody);

//                Tuple2<Double, Double> result = resultDataset.get(0);
//                double loss = result._1 / batchNum;
//                double accuracy = result._2 / batchNum;
//                System.out.format("\tEpoch: %d, Loss: %.4f, Acc: %.4f\n", i+1, loss, accuracy);




        DataSet<Integer> temp = finalDataset.map((batch) -> 1)
                .reduce((cnt1, cnt2) -> cnt1 + cnt2);
        temp.print();
    }
}
