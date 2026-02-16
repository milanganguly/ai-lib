package milanganguly.engine;

import java.util.Arrays;

public class Tensor {
    private float[] data;
    private float[] grad;
    private final int[] shape;

    private final boolean requiresGrad;

    Tensor[] parents;
    Operation backwardsOp = null;

    public Tensor(float[] data, boolean requiresGrad, int... shape) {
        this.data = data;
        this.requiresGrad = requiresGrad;
        this.shape = shape;
        this.grad = new float[numel()];
        if (data.length != numel()) {
            throw new IllegalArgumentException(
                    "Data length does not match shape"
            );
        }
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
    public float[] getData() {
        return data.clone();
    }
    public float[] getGrad() {
        return grad.clone();
    }
    public int[] getShape() {
        return shape.clone();
    }
    public void addScaledGrad(float scale) {
        for (int i = 0; i < numel(); i++) {
            data[i] += grad[i]*scale;
        }
    }
    void accumulateGrad(float[] incomingGrad) {
        if (incomingGrad.length != numel()) {
            throw new IllegalArgumentException("Incoming grad array lengths do not match tensor size");
        }
        for (int i = 0; i < incomingGrad.length; i++) {
            grad[i] += incomingGrad[i];
        }
    }
    public boolean requiresGrad() {
        return requiresGrad;
    }

    void fillGrad(float value) {
        Arrays.fill(this.grad, value);
    }
    public void addScaledUpdate(float[] update, float scale) {
        for (int i = 0; i < numel(); i++) {
            data[i] += update[i]*scale;
        }
    }
    float[] gradRef() {
        return grad;
    }
    float[] dataRef() {
        return data;
    }
}
