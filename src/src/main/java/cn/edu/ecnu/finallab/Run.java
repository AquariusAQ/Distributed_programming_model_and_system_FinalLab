package cn.edu.ecnu.finallab;

import cn.edu.ecnu.finallab.benchmark.RunKmeans;
import cn.edu.ecnu.finallab.benchmark.RunMLP;
import cn.edu.ecnu.finallab.benchmark.RunPCA;

public class Run {
    public static void main(String[] args) throws Exception {
//        String[] test = new String[]{"PCA", "spark"};
        // 0: MLP, Kmeans, PCA
        // 1: spark, flink, flink-bulk
        // 2: vanilla, broadcast, no-broadcast, communication
        // 3: epoch
        // 4: input image
        // 5: input label
        // 6: warmUpRound
        // 7: testRound
        int warmUpRound = Integer.parseInt(args[6]);
        int testRound = Integer.parseInt(args[7]);
        if (testRound < 1) {
            System.out.println("testRount must be greater than 0!");
        }
        
        double[] allTime = new double[warmUpRound + testRound];

        System.out.println("---------- Start test ----------");
        System.out.print("args: ");
        for (String arg : args) {
            System.out.print(arg + " ");
        }
        System.out.println();

        double usedTime = 0.;
        for (int i = 0; i < warmUpRound + testRound; i++) {
            if (i < warmUpRound) {
                System.out.println("------- Warming up " + (i+1) + " --------");
            } else {
                System.out.println("---------- Round " + (i-warmUpRound+1) + " ----------");
            }
            if (args[0].equals("MLP")) {
                usedTime = RunMLP.run(args);
            } else if (args[0].equals("Kmeans")) {
                usedTime = RunKmeans.run(args);
            } else if (args[0].equals("PCA")) {
                usedTime = RunPCA.run(args);
            } else {
                System.out.println("Invalid job name!");
            }
            System.out.format("Used time: %.2f sec.\n", usedTime);
            allTime[i] = usedTime;
        }
        System.out.println("---------- Summary ----------");
        String title = "| ";
        String value = "|  ";

        double timeSum = 0.;
        for (int i = 0; i < warmUpRound + testRound; i++) {
            if (i < warmUpRound) {
                title += " Warm up " + (i+1) + " |";
            } else {
                title += " Round " + (i-warmUpRound+1) + " |";
                timeSum += allTime[i];
            }
            value += "  " + allTime[i] + "  |";
        }
        System.out.println(title);
        System.out.println(value);
        System.out.println("Avg time: " + (timeSum / (testRound)) + " sec.");
    }
}
