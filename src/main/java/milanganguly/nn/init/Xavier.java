package milanganguly.nn.init;

import milanganguly.engine.Tensor;

import java.util.Random;

class Xavier implements Initializer {

    private final Random rand = new Random();

    @Override
    public Tensor initialize(boolean requiresGrad, int... shape) {
        int fanIn, fanOut;

        if (shape.length == 2) {        // for linears
            fanIn = shape[0];
            fanOut = shape[1];
        } else if (shape.length == 4) {     // for conv2d
            fanIn = shape[1] * shape[2] * shape[3];
            fanOut = shape[0] * shape[2] * shape[3];
        } else {            // um assume fully connected otherwise
            fanIn = shape[0];
            fanOut = shape[1];
        }
        float limit = (float) Math.sqrt(2.0 / (fanIn + fanOut));
        int size = 1;
        for (int dim : shape) size *= dim;
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = (rand.nextFloat() * 2f - 1f) * limit;
        }
        return new Tensor(data, requiresGrad, shape);
    }
}
