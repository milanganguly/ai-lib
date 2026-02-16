import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;

public class MNISTLoader {

    public static class MNISTData {
        public final float[][] images;
        public final int[] labels;

        public MNISTData(float[][] images, int[] labels) {
            this.images = images;
            this.labels = labels;
        }
    }

    public static MNISTData loadTrain() throws IOException {
        return load("train-images-idx3-ubyte",
                "train-labels-idx1-ubyte");
    }

    public static MNISTData loadTest() throws IOException {
        return load("t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte");
    }


    private static MNISTData load(String imageFile, String labelFile) throws IOException {

        try (DataInputStream images = open(imageFile);
             DataInputStream labels = open(labelFile)) {

            // ----- Images header -----
            int imageMagic = images.readInt();
            int numImages  = images.readInt();
            int rows       = images.readInt();
            int cols       = images.readInt();

            // ----- Labels header -----
            int labelMagic = labels.readInt();
            int numLabels  = labels.readInt();

            if (numImages != numLabels) {
                throw new RuntimeException("Image count does not match label count.");
            }

            float[][] imageData = new float[numImages][rows * cols];
            int[] labelData = new int[numImages];

            for (int i = 0; i < numImages; i++) {

                for (int j = 0; j < rows * cols; j++) {
                    int pixel = images.readUnsignedByte();
                    imageData[i][j] = pixel / 255.0f;
                }

                labelData[i] = labels.readUnsignedByte();
            }

            return new MNISTData(imageData, labelData);
        }
    }

    private static DataInputStream open(String name) {

        InputStream is = MNISTLoader.class
                .getClassLoader()
                .getResourceAsStream("mnist/" + name);

        if (is == null) {
            throw new RuntimeException("Could not find MNIST file: " + name);
        }

        return new DataInputStream(new BufferedInputStream(is));
    }
}
