package milanganguly.engine;

import java.util.Arrays;

public class TensorOps {
    public static Tensor add(Tensor a, Tensor b) {
        if (!Arrays.equals(a.getShape(), b.getShape())) {
            throw new IllegalArgumentException("Tensor sizes must match for add.");
        }
        float[] aData = a.getData();
        float[] bData = b.getData();
        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            outData[i] = aData[i]+ bData[i];
        }
        Tensor out;
        if (a.requiresGrad() || b.requiresGrad()) {
            out = new Tensor(outData, true, a.getShape());
            out.parents = new Tensor[] {a, b};
            out.backwardsOp = Autograd.addBackward(a, b) ;
        } else {
            out = new Tensor(outData, a.getShape());
        }
        return out;
    }
    public static Tensor mul(Tensor a, Tensor b) {
        if (a.numel()!=b.numel()) {
            throw new IllegalArgumentException("Tensor sizes must match for mul.");
        }
        float[] aData = a.getData();
        float[] bData = b.getData();

        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            outData[i] = aData[i]* bData[i];
        }
        Tensor out;
        if (a.requiresGrad() || b.requiresGrad()) {
            out = new Tensor(outData, true, a.getShape());
            out.parents = new Tensor[] {a, b};
            out.backwardsOp = Autograd.mulBackward(a, b);
        } else {
            out  = new Tensor(outData, a.getShape());
        }
        return out;
    }
    public static Tensor neg(Tensor a) {
        float[] aData = a.getData();

        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            outData[i] = -aData[i];
        }
        Tensor out;
        if (a.requiresGrad()) {
            out = new Tensor(outData, true, a.getShape());
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.negBackward(a);
        } else {
             out = new Tensor(outData, a.getShape());
        }
        return out;
    }
    public static Tensor matmul(Tensor a, Tensor b) {
        float[] aData = a.getData();
        float[] bData = b.getData();

        int[] aShape = a.getShape();
        int[] bShape = b.getShape();

        if (!(a.getShape().length == b.getShape().length && a.getShape().length == 2)) {
            throw new IllegalArgumentException("matmul requires 2 dimensional tensors.");
        }
        int m = aShape[0];   // rows of A
        int k = aShape[1];   // cols of A
        int k2 = bShape[0];  // rows of B
        int n = bShape[1];   // cols of B

        if (k != k2) {
            throw new IllegalArgumentException("Tensor shapes do not match for matmul.");
        }
        float[] outData = new float[m*n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0f;
                for (int p = 0; p < k; p++) {
                    float aVal = aData[i * k + p];   // A[i,p]
                    float bVal = bData[p * n + j];   // B[p,j]
                    sum += aVal * bVal;
                }
                outData[i * n + j] = sum; // C[i,j]
            }
        }
        Tensor out;
        if (a.requiresGrad() || b.requiresGrad()) {
            out = new Tensor(outData, true, m, n);
            out.parents = new Tensor[] {a, b};
            out.backwardsOp = Autograd.matMulBackward(a, b);
        } else {
            out = new Tensor(outData, m, n);
        }
        return out;
    }
    public static Tensor ReLU(Tensor a) {
        float[] aData = a.getData();

        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            outData[i] = Math.max(aData[i], 0);
        }
        Tensor out;
        if (a.requiresGrad()) {
            out = new Tensor(outData, true, a.getShape());
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.ReLUBackward(a);
        } else {
            out = new Tensor(outData, a.getShape());
        }
        return out;
    }
    public static Tensor sum(Tensor a) {
        float[] aData = a.getData();

        float[] sum = new float[] {0f};
        for (int i = 0; i < a.numel(); i++) {
            sum[0] += aData[i];
        }
        Tensor out;
        if (a.requiresGrad()) {
            out = new Tensor(sum, true, new int[] {});
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.sumBackward(a);
        } else {
            out = new Tensor(sum, new int[] {});
        }
        return out;
    }
    public static Tensor mean(Tensor a) {
        Tensor s = sum(a);
        float scale = 1f / a.numel();
        return mul(s, new Tensor(new float[]{scale}, new int[]{}));
    }
    public static Tensor mse(Tensor pred, Tensor target) {
        Tensor diff = add(target, neg(pred));
        Tensor sq = mul(diff, diff);
        return mean(sq);
    }
    public static Tensor mae(Tensor pred, Tensor target) {
        Tensor diff = add(target, neg(pred));
        return mean(diff);
    }
    public static Tensor exp(Tensor a) {
        float[] aData = a.getData();

        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            outData[i] = (float) Math.exp(aData[i]);
        }
        Tensor out;
        if (a.requiresGrad()) {
            out = new Tensor(outData, true, a.getShape());
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.expBackward(out);
        } else {
            out = new Tensor(outData, a.getShape());
        }
        return out;
    }
    public static Tensor sigmoid(Tensor a) {
        float[] aData = a.getData();

        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            float x = aData[i];
            outData[i] = (float)(1.0 / (1.0 + Math.exp(-x)));
        }
        Tensor out;

        if (a.requiresGrad()) {
            out = new Tensor(outData, true, a.getShape());
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.sigmoidBackward(out);
        } else {
            out = new Tensor(outData, a.getShape());
        }
        return out;
    }
    public static Tensor flatten(Tensor a) {

        Tensor out;
        if (a.requiresGrad()) {
            out = new Tensor(a.dataRef(), true, a.numel());
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.temp(a);
        } else {
            out = new Tensor(a.dataRef(), a.numel());
        }
        return out;
    }
    public static Tensor reshape(Tensor a, int[] newShape) {
        int elts = 1;
        for (int j : newShape) {
            elts *= j;
        }
        if (elts != a.numel()) {
            throw new IllegalArgumentException("Tensor size does not match the reshape.");
        }
        Tensor out;
        if (a.requiresGrad()) {
            out = new Tensor(a.dataRef(), true, newShape);
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.reshapeBackward(a);
        } else {
            out = new Tensor(a.dataRef(), newShape);
        }
        return out;
    }
    public static Tensor transpose(Tensor a) {

        int[] aShape = a.getShape();
        if (aShape.length != 2) {
            throw new IllegalArgumentException("transpose requires a 2D tensor.");
        }

        float[] aData = a.getData();

        int m = aShape[0];
        int n = aShape[1];
        float[] outData = new float[a.numel()];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                outData[j * m + i] = aData[i * n + j];
            }
        }
        Tensor out;

        if (a.requiresGrad()) {
            out = new Tensor(outData, true, new int[] { n, m });
            out.parents = new Tensor[] { a };
            out.backwardsOp = Autograd.transposeBackward(a);
        } else {
            out = new Tensor(outData, new int[] { n, m });
        }

        return out;
    }
    public static Tensor addBias(Tensor a, Tensor b) {
        float[] aData = a.getData();
        float[] bData = b.getData();

        int[] aShape = a.getShape();
        int[] bShape = b.getShape();
        if (aShape.length != 2) {
            throw new IllegalArgumentException("First tensor dimension must be 2 for addBias.");
        }
        if (bShape.length != 1) {
            throw new IllegalArgumentException("Second tensor dimension must be 1 for addBias.");
        }
        if (bShape[0] != aShape[1]) {
            throw new IllegalArgumentException("Tensor shapes don't match for addBias.");
        }
        float[] outData = new float[a.numel()];
        for (int i = 0; i < aShape[0]; i++) {
            for (int j = 0; j < aShape[1]; j++) {
                outData[i* aShape[1] + j] = aData[i* aShape[1] + j] + bData[j];
            }
        }
        Tensor out;
        if (a.requiresGrad() || b.requiresGrad()) {
            out = new Tensor(outData, true, a.getShape());
            out.parents = new Tensor[] {a, b};
            out.backwardsOp = Autograd.addBiasBackward(a, b);
        } else {
            out = new Tensor(outData, a.getShape());
        }
        return out;
    }

    public static Tensor im2col(Tensor input, int kernelH, int kernelW, int stride, int padding) {
        int[] shape = input.getShape();
        int C = shape[0];
        int H = shape[1];
        int W = shape[2];
        int H_out = (H - kernelH + 2 * padding) / stride + 1;
        int W_out = (W - kernelW + 2 * padding) / stride + 1;
        int rows = H_out * W_out;
        int cols = C * kernelH * kernelW;
        float[] outData = new float[rows * cols];
        float[] inData = input.getData();
        int rowIndex = 0;
        for (int outY = 0; outY < H_out; outY++) {
            for (int outX = 0; outX < W_out; outX++) {
                int colIndex = 0;
                for (int c = 0; c < C; c++) {
                    for (int kY = 0; kY < kernelH; kY++) {
                        for (int kX = 0; kX < kernelW; kX++) {
                            int inY = outY * stride + kY - padding;
                            int inX = outX * stride + kX - padding;
                            float value = 0f;
                            if (inY >= 0 && inY < H &&
                                    inX >= 0 && inX < W) {
                                int inputIndex = c * H * W + inY * W + inX;
                                value = inData[inputIndex];
                            }
                            int outputIndex = rowIndex * cols + colIndex;
                            outData[outputIndex] = value;
                            colIndex++;
                        }
                    }
                }
                rowIndex++;
            }
        }
        Tensor out;
        if (input.requiresGrad()) {
            out = new Tensor(outData, true, new int[] { rows, cols });
            out.parents = new Tensor[] { input };
            out.backwardsOp = Autograd.im2colBackward(input, kernelH, kernelW, stride, padding, H_out, W_out);
        } else {
            out = new Tensor(outData, new int[] { rows, cols });
        }
        return out;
    }
    public static Tensor abs(Tensor a) {
        float[] aData = a.getData();

        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            outData[i] = Math.abs(aData[i]);
        }
        Tensor out;
        if (a.requiresGrad()) {
            out = new Tensor(outData, true, a.getShape());
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.absBackward(a);
        } else {
            out = new Tensor(outData, a.getShape());
        }
        return out;
    }
    public static Tensor sumRows(Tensor a) {
        int[] shape = a.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("sumRows requires 2D tensor.");
        }
        int rows = shape[0];
        int cols = shape[1];
        float[] aData = a.dataRef();
        float[] outData = new float[rows];
        for (int i = 0; i < rows; i++) {
            float sum = 0f;
            for (int j = 0; j < cols; j++) {
                sum += aData[i * cols + j];
            }
            outData[i] = sum;
        }
        Tensor out;
        if (a.requiresGrad()) {
            out = new Tensor(outData, true, rows);
            out.parents = new Tensor[]{a};
            out.backwardsOp = Autograd.sumRowsBackward(a);
        } else {
            out = new Tensor(outData, rows);
        }
        return out;
    }
    public static Tensor maxRows(Tensor a) {
        int[] shape = a.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("maxRows requires 2D tensor.");
        }
        int rows = shape[0];
        int cols = shape[1];
        float[] aData = a.dataRef();
        float[] outData = new float[rows];
        for (int i = 0; i < rows; i++) {
            float max = aData[i * cols];
            for (int j = 1; j < cols; j++) {
                float val = aData[i * cols + j];
                if (val > max) max = val;
            }
            outData[i] = max;
        }
        Tensor out;
        if (a.requiresGrad()) {
            out = new Tensor(outData, true, rows);
            out.parents = new Tensor[]{a};
            out.backwardsOp = Autograd.maxRowsBackward(a, out);
        } else {
            out = new Tensor(outData, rows);
        }

        return out;
    }
    public static Tensor log(Tensor a) {
        float[] aData = a.dataRef();
        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            outData[i] = (float)Math.log(aData[i]);
        }
        Tensor out;
        if (a.requiresGrad()) {
            out = new Tensor(outData, true, a.getShape());
            out.parents = new Tensor[]{a};
            out.backwardsOp = Autograd.logBackward(a);
        } else {
            out = new Tensor(outData, a.getShape());
        }
        return out;
    }
    public static Tensor subtractRowVector(Tensor a, Tensor rowVec) {
        int[] shape = a.getShape();
        if (shape.length != 2)
            throw new IllegalArgumentException("subtractRowVector requires 2D tensor.");

        int rows = shape[0];
        int cols = shape[1];

        if (rowVec.getShape().length != 1 || rowVec.getShape()[0] != rows)
            throw new IllegalArgumentException("Row vector must match rows.");

        float[] aData = a.dataRef();
        float[] vData = rowVec.dataRef();
        float[] outData = new float[a.numel()];

        for (int i = 0; i < rows; i++) {
            float v = vData[i];
            for (int j = 0; j < cols; j++) {
                outData[i * cols + j] = aData[i * cols + j] - v;
            }
        }

        Tensor out = new Tensor(outData, a.requiresGrad(), shape);

        if (a.requiresGrad()) {
            out.parents = new Tensor[]{a, rowVec};
            out.backwardsOp = Autograd.subtractRowBackward(a, rowVec);
        }

        return out;
    }
    public static Tensor crossEntropy(Tensor logits, Tensor target) {

        // subtract row-wise max, better stability
        Tensor max = maxRows(logits);
        Tensor stabilized = subtractRowVector(logits, max);

        Tensor exp = exp(stabilized);
        Tensor sumExp = sumRows(exp);
        Tensor logSumExp = log(sumExp);

        Tensor logSoftmax = subtractRowVector(stabilized, logSumExp);

        Tensor mul = mul(target, logSoftmax);

        Tensor total = sum(mul);

        Tensor neg = neg(total);

        float scale = 1f / logits.getShape()[0];
        Tensor scaleTensor = new Tensor(new float[]{scale}, new int[]{});
        return mul(neg, scaleTensor);
    }
    public static Tensor maxPool2D(Tensor input, int poolH, int poolW, int stride) {
        int[] shape = input.getShape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("maxPool2D expects [C, H, W]");
        }
        int C = shape[0];
        int H = shape[1];
        int W = shape[2];
        int H_out = (H - poolH) / stride + 1;
        int W_out = (W - poolW) / stride + 1;
        float[] inData = input.dataRef();
        float[] outData = new float[C * H_out * W_out];
        for (int c = 0; c < C; c++) {
            for (int outY = 0; outY < H_out; outY++) {
                for (int outX = 0; outX < W_out; outX++) {
                    float max = Float.NEGATIVE_INFINITY;
                    for (int kY = 0; kY < poolH; kY++) {
                        for (int kX = 0; kX < poolW; kX++) {
                            int inY = outY * stride + kY;
                            int inX = outX * stride + kX;
                            int index = c * H * W + inY * W + inX;
                            float val = inData[index];
                            if (val > max) max = val;
                        }
                    }
                    int outIndex = c * H_out * W_out + outY * W_out + outX;
                    outData[outIndex] = max;
                }
            }
        }
        Tensor out;
        if (input.requiresGrad()) {
            out = new Tensor(outData, true, new int[]{C, H_out, W_out});
            out.parents = new Tensor[]{input};
            out.backwardsOp =
                    Autograd.maxPool2DBackward(input, poolH, poolW, stride, H_out, W_out);
        } else {
            out = new Tensor(outData, new int[]{C, H_out, W_out});
        }
        return out;
    }


}
