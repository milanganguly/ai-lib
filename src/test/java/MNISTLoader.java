import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MNISTLoader {

    public static class MNISTData {
        public float[][] images;
        public int[] labels;

        public MNISTData(float[][] images, int[] labels) {
            this.images = images;
            this.labels = labels;
        }
    }

    public static MNISTData loadTrain() throws IOException {
        return load(
                "/mnist/train-images-idx3-ubyte",
                "/mnist/train-labels-idx1-ubyte"
        );
    }

    public static MNISTData loadTest() throws IOException {
        return load(
                "/mnist/t10k-images-idx3-ubyte",
                "/mnist/t10k-labels-idx1-ubyte"
        );
    }

    private static MNISTData load(String imagePath, String labelPath) throws IOException {

        InputStream imageStream = MNISTLoader.class.getResourceAsStream(imagePath);
        InputStream labelStream = MNISTLoader.class.getResourceAsStream(labelPath);

        DataInputStream images = new DataInputStream(new BufferedInputStream(imageStream));
        DataInputStream labels = new DataInputStream(new BufferedInputStream(labelStream));

        // ---- Read image header ----
        images.readInt(); // magic number
        int numImages = images.readInt();
        int rows = images.readInt();
        int cols = images.readInt();

        // ---- Read label header ----
        labels.readInt(); // magic number
        int numLabels = labels.readInt();

        if (numImages != numLabels)
            throw new IllegalStateException("Mismatch between images and labels");

        float[][] imageData = new float[numImages][rows * cols];
        int[] labelData = new int[numLabels];

        for (int i = 0; i < numImages; i++) {
            for (int j = 0; j < rows * cols; j++) {
                int pixel = images.readUnsignedByte();
                imageData[i][j] = pixel / 255.0f; // normalize
            }
            labelData[i] = labels.readUnsignedByte();
        }

        images.close();
        labels.close();

        return new MNISTData(imageData, labelData);
    }
}
