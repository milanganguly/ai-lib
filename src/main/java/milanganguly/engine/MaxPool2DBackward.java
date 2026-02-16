package milanganguly.engine;

class MaxPool2DBackward implements Operation {

    private final Tensor input;
    private final int poolH, poolW, stride;
    private final int H_out, W_out;

    public MaxPool2DBackward(Tensor input, int poolH, int poolW, int stride, int H_out, int W_out) {
        this.input = input;
        this.poolH = poolH;
        this.poolW = poolW;
        this.stride = stride;
        this.H_out = H_out;
        this.W_out = W_out;
    }

    @Override
    public void backward(float[] gradOutput) {
        if (!input.requiresGrad()) return;
        int[] shape = input.getShape();
        int C = shape[0];
        int H = shape[1];
        int W = shape[2];
        float[] inData = input.dataRef();
        float[] inGrad = input.gradRef();
        for (int c = 0; c < C; c++) {
            for (int outY = 0; outY < H_out; outY++) {
                for (int outX = 0; outX < W_out; outX++) {
                    int outIndex = c * H_out * W_out + outY * W_out + outX;
                    float grad = gradOutput[outIndex];
                    float max = Float.NEGATIVE_INFINITY;
                    int maxIndex = -1;
                    for (int kY = 0; kY < poolH; kY++) {
                        for (int kX = 0; kX < poolW; kX++) {
                            int inY = outY * stride + kY;
                            int inX = outX * stride + kX;
                            int index = c * H * W + inY * W + inX;
                            float val = inData[index];
                            if (val > max) {
                                max = val;
                                maxIndex = index;
                            }
                        }
                    }
                    inGrad[maxIndex] += grad;
                }
            }
        }
    }
}

