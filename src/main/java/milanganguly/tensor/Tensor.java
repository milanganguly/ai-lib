package milanganguly.tensor;

import milanganguly.autograd.Operation;

public class Tensor {
    public float[] data;
    public float[] grad;
    public int[] shape;

    public boolean requiresGrad = false;

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
        for (int i = 0; i<this.grad.length; i++) {
            this.grad[i] = 0;
        }
    }

}
