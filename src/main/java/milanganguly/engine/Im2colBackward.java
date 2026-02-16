package milanganguly.engine;

public class Im2colBackward implements Operation {

    private final Tensor input;
    private final int kernelH, kernelW;
    private final int stride, padding;
    private final int H_out, W_out;

    public Im2colBackward(Tensor input, int kernelH, int kernelW, int stride, int padding, int H_out, int W_out) {
        this.input = input;
        this.kernelH = kernelH;
        this.kernelW = kernelW;
        this.stride = stride;
        this.padding = padding;
        this.H_out = H_out;
        this.W_out = W_out;
    }

    @Override
    public void backward(float[] gradOutput) {
        if (input.requiresGrad()) {
            int[] shape = input.getShape();
            int C = shape[0];
            int H = shape[1];
            int W = shape[2];
            int cols = C * kernelH * kernelW;
            float[] inputGrad = input.gradRef();
            int rowIndex = 0;
            for (int outY = 0; outY < H_out; outY++) {
                for (int outX = 0; outX < W_out; outX++) {
                    int colIndex = 0;
                    for (int c = 0; c < C; c++) {
                        for (int kY = 0; kY < kernelH; kY++) {
                            for (int kX = 0; kX < kernelW; kX++) {
                                int inY = outY * stride + kY - padding;
                                int inX = outX * stride + kX - padding;
                                if (inY >= 0 && inY < H && inX >= 0 && inX < W) {
                                    int inputIndex = c * H * W + inY * W + inX;
                                    int outputIndex = rowIndex * cols + colIndex;
                                    inputGrad[inputIndex] += gradOutput[outputIndex];
                                }
                                colIndex++;
                            }
                        }
                    }
                    rowIndex++;
                }
            }
        }
    }
}
