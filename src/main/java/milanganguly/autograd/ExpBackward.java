package milanganguly.autograd;

import milanganguly.tensor.Tensor;

class ExpBackward implements Operation {
    private final Tensor a;

    public ExpBackward(Tensor output) {
        this.a = output;
    }

    @Override
    public void backward(float[] gradOutput) {
        if (a.requiresGrad) {
            int size = a.data.length;
            for (int i = 0; i < size; i++) {
                a.grad[i] += gradOutput[i] * a.data[i];
            }
        }
    }
}
