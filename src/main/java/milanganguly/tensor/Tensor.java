package milanganguly.tensor;

import milanganguly.autograd.Operation;

import java.util.Arrays;

public class Tensor {
    public float[] data;
    public float[] grad;
    public int[] shape;

    public boolean requiresGrad;

    public Tensor[] parents;
    public Operation backwardsOp = null;

    public Tensor(float[] data, boolean requiresGrad, int... shape) {
        this.data = data;
        this.requiresGrad = requiresGrad;
        this.shape = shape;
        this.grad = new float[numel()];
    }
    public Tensor(float[] data, int... shape) {
        this(data, false, shape);
    }
    public int numel() {
        int n = 1;
        for (int s : shape) n *= s;
        return n;
    }
    public void zeroGrad() {
        Arrays.fill(this.grad, 0);
    }
}
