package milanganguly.tensor;

import milanganguly.autograd.Autograd;

import java.util.Arrays;

public class TensorOps {
    public static Tensor add(Tensor a, Tensor b) {
        if (!Arrays.equals(a.shape, b.shape)) {
            throw new IllegalArgumentException("Tensor sizes must match for add.");
        }
        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            outData[i] = a.data[i]+b.data[i];
        }
        Tensor out = new Tensor(outData, a.shape);
        if (a.requiresGrad || b.requiresGrad) {
            out.requiresGrad = true;
            out.parents = new Tensor[] {a, b};
            out.backwardsOp = Autograd.addBackward(a, b) ;
        }
        return out;
    }
    public static Tensor mul(Tensor a, Tensor b) {
        if (a.numel()!=b.numel()) {
            throw new IllegalArgumentException("Tensor sizes must match for mul.");
        }
        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            outData[i] = a.data[i]*b.data[i];
        }
        Tensor out = new Tensor(outData, a.shape);
        if (a.requiresGrad || b.requiresGrad) {
            out.requiresGrad = true;
            out.parents = new Tensor[] {a, b};
            out.backwardsOp = Autograd.mulBackward(a, b);
        }
        return out;
    }
    public static Tensor neg(Tensor a) {
        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            outData[i] = -a.data[i];
        }
        Tensor out = new Tensor(outData, a.shape);
        if (a.requiresGrad) {
            out.requiresGrad = true;
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.negBackward(a);
        }
        return out;
    }
    public static Tensor matmul(Tensor a, Tensor b) {
        if (!(a.shape.length == b.shape.length && a.shape.length == 2)) {
            throw new IllegalArgumentException("matmul requires 2 dimensional tensors.");
        }
        int m = a.shape[0];   // rows of A
        int k = a.shape[1];   // cols of A
        int k2 = b.shape[0];  // rows of B
        int n = b.shape[1];   // cols of B

        if (k != k2) {
            throw new IllegalArgumentException("Tensor shapes do not match for matmul.");
        }
        float[] outData = new float[m*n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0f;
                for (int p = 0; p < k; p++) {
                    float aVal = a.data[i * k + p];   // A[i,p]
                    float bVal = b.data[p * n + j];   // B[p,j]
                    sum += aVal * bVal;
                }
                outData[i * n + j] = sum; // C[i,j]
            }
        }
        Tensor out = new Tensor(outData, m, n);
        if (a.requiresGrad || b.requiresGrad) {
            out.requiresGrad = true;
            out.parents = new Tensor[] {a, b};
            out.backwardsOp = Autograd.matMulBackward(a, b);
        }
        return out;
    }
    public static Tensor ReLU(Tensor a) {
        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            outData[i] = Math.max(a.data[i], 0);
        }
        Tensor out = new Tensor(outData, a.shape);
        if (a.requiresGrad) {
            out.requiresGrad = true;
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.ReLUBackward(a);
        }
        return out;
    }
    public static Tensor sum(Tensor a) {
        float[] sum = new float[] {0f};
        for (int i = 0; i < a.numel(); i++) {
            sum[0] += a.data[i];
        }
        Tensor out = new Tensor(sum, new int[] {});
        if (a.requiresGrad) {
            out.requiresGrad = true;
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.sumBackward(a);
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
    public static Tensor exp(Tensor a) {
        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            outData[i] = (float) Math.exp(a.data[i]);
        }
        Tensor out = new Tensor(outData, a.shape);
        if (a.requiresGrad) {
            out.requiresGrad = true;
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.expBackward(a);
        }
        return out;
    }
    public static Tensor sigmoid(Tensor a) {
        float[] outData = new float[a.numel()];
        for (int i = 0; i < a.numel(); i++) {
            float x = a.data[i];
            outData[i] = (float)(1.0 / (1.0 + Math.exp(-x)));
        }

        Tensor out = new Tensor(outData, a.shape.clone());

        if (a.requiresGrad) {
            out.requiresGrad = true;
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.sigmoidBackward(out);
        }
        return out;
    }
    public static Tensor reshape(Tensor a, int[] newShape) {
        int elts = 1;
        for (int j : newShape) {
            elts *= j;
        }
        if (elts != a.numel()) {
            throw new IllegalArgumentException("Tensor shapes do not match for reshape.");
        }
        Tensor out = new Tensor(a.data, newShape);
        if (a.requiresGrad) {
            out.requiresGrad = true;
            out.parents = new Tensor[] {a};
            out.backwardsOp = Autograd.reshapeBackward(a);
        }
        return out;
    }
    public static Tensor transpose(Tensor a) {
        int m = a.shape[0];
        int n = a.shape[1];
        float[] outData = new float[a.numel()];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                outData[j * m + i] = a.data[i * n + j];
            }
        }
        Tensor out = new Tensor(outData, new int[] { n, m });

        if (a.requiresGrad) {
            out.requiresGrad = true;
            out.parents = new Tensor[] { a };
            out.backwardsOp = Autograd.transposeBackward(a);
        }

        return out;
    }

}
