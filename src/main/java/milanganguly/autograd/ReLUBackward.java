package milanganguly.autograd;

import milanganguly.tensor.Tensor;

class ReLUBackward implements Operation {
    private final Tensor a;
    public ReLUBackward(Tensor a) {
        this.a = a;
    }

    @Override
    public void backward(float[] gradOutput) {
        for (int i = 0; i < a.grad.length; i++) {
            if (a.data[i] > 0) {
                a.grad[i] += gradOutput[i];
            }
        }
    }
}
