
import milanganguly.engine.Autograd;
import milanganguly.engine.Tensor;
import milanganguly.nn.*;
import milanganguly.loss.CrossEntropy;
import milanganguly.nn.layers.*;
import milanganguly.optim.Adam;
import milanganguly.optim.SGD;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class Test {

    public static void main(String[] args) throws IOException {

        // ---- Load MNIST here ----
        MNISTLoader.MNISTData data = MNISTLoader.loadTrain();
        float[][] images = data.images;
        int[] labels = data.labels;


        // ---- Build Model ----
        Sequential model = new Sequential(
                new Conv2D(1, 8, 3, 3),
                new ReLU(),
                new MaxPool2D(2, 2, 2),

                new Conv2D(8, 16, 3, 3),
                new ReLU(),
                new MaxPool2D(2, 2, 2),

                new Flatten(),
                new Linear(16 * 5 * 5, 32),
                new ReLU(),
                new Linear(32, 32),
                new ReLU(),
                new Linear(32, 32),
                new ReLU(),
                new Linear(32, 10)
        );
        /*
        Sequential model = new Sequential(
                new Flatten(),
                new Linear(28 * 28, 128),
                new ReLU(),
                new Linear(128, 10)
        );
        */
        CrossEntropy lossFn = new CrossEntropy();
        Adam optimizer = new Adam(model, 0.001f);        // sgd was 50% at 1700, Adam 50% by 600

        int epochs = 100;
        Random rand = new Random();
        System.out.println("Starting training...");

        for (int epoch = 0; epoch < epochs; epoch++) {

            float totalLoss = 0f;
            int correct = 0;

            int limit = images.length;

            for (int i = 0; i < limit; i++) {


                // ---- Prepare Input ----
                float[] img = images[i];
                int label = labels[i];

                Tensor input = new Tensor(img, true, 1, 28, 28);

                Tensor target = oneHot(label, 10);

                // ---- Forward ----
                Tensor logits = model.forward(input);
                Tensor loss = lossFn.forward(logits, target);

                totalLoss += loss.getData()[0];

                // ---- Accuracy ----
                int pred = argmax(logits);
                if (pred == label) correct++;

                // ---- Backward ----
                optimizer.zeroGrad();
                Autograd.backward(loss);
                optimizer.step();

                if (i % 100 == 0) {
                    System.out.println("Sample " + i);
                    System.out.println("Accuracy " + (float) correct/(i+1));
                }

            }

            float avgLoss = totalLoss / limit;
            float acc = (float) correct / limit;

            System.out.println("Epoch " + epoch +
                    " | Loss: " + avgLoss +
                    " | Accuracy: " + acc);
            if (avgLoss<0.001f || acc > 0.9f) {
                return;
            }
        }
    }

    private static Tensor oneHot(int label, int numClasses) {
        float[] vec = new float[numClasses];
        vec[label] = 1f;
        return new Tensor(vec, 1, numClasses);
    }

    private static int argmax(Tensor logits) {
        float[] data = logits.getData();
        int maxIdx = 0;
        float maxVal = data[0];

        for (int i = 1; i < data.length; i++) {
            if (data[i] > maxVal) {
                maxVal = data[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }
}
