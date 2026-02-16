import milanganguly.engine.Tensor;
import milanganguly.engine.Autograd;
import milanganguly.nn.*;
import milanganguly.loss.CrossEntropy;
import milanganguly.optim.Adam;
import milanganguly.nn.layers.*;

public class MNISTDemo {

    public static void main(String[] args) throws Exception {

        System.out.println("Loading MNIST...");
        MNISTLoader.MNISTData train = MNISTLoader.loadTrain();
        MNISTLoader.MNISTData test  = MNISTLoader.loadTest();

        Sequential model = new Sequential(new milanganguly.nn.layers.Conv2D(1, 8, 3, 3), new ReLU(), new MaxPool2D(2, 2, 2), new Conv2D(8, 16, 3, 3), new ReLU(), new MaxPool2D(2, 2, 2), new Flatten(), new Linear(16 * 5 * 5, 32), new ReLU(), new Linear(32, 10));

        CrossEntropy lossFn = new CrossEntropy();
        Adam optimizer = new Adam(model, 0.001f);

        int epochs = 5;

        System.out.println("Starting training...\n");

        for (int epoch = 0; epoch < epochs; epoch++) {

            float trainLoss = 0f;
            int correct = 0;

            for (int i = 0; i < train.images.length; i++) {

                Tensor input = new Tensor(train.images[i], true, 1, 28, 28);
                Tensor target = oneHot(train.labels[i], 10);

                Tensor logits = model.forward(input);
                Tensor loss = lossFn.forward(logits, target);

                trainLoss += loss.getData()[0];

                int pred = argmax(logits);
                if (pred == train.labels[i]) correct++;

                optimizer.zeroGrad();
                Autograd.backward(loss);
                optimizer.step();
            }

            float avgLoss = trainLoss / train.images.length;
            float trainAcc = (float) correct / train.images.length;

            float testAcc = evaluate(model, test.images, test.labels);

            System.out.printf("Epoch %d | Loss: %.4f | Train Acc: %.4f | Test Acc: %.4f\n", epoch + 1, avgLoss, trainAcc, testAcc);
        }

        System.out.println("\nTraining complete.");
    }

    private static float evaluate(Sequential model, float[][] images, int[] labels) {
        int correct = 0;
        for (int i = 0; i < images.length; i++) {
            Tensor input = new Tensor(images[i], false, 1, 28, 28);
            Tensor logits = model.forward(input);
            int pred = argmax(logits);
            if (pred == labels[i]) correct++;
        }
        return (float) correct / images.length;
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
