package milanganguly.autograd;

import milanganguly.tensor.Tensor;

class ReshapeBackward implements Operation {
    private final Tensor a;

    public ReshapeBackward(Tensor a) {
        this.a = a;
    }

    @Override
    public void backward(float[] gradOutput) {
        if (a.requiresGrad) {
            for (int i = 0; i < a.numel(); i++) {
                a.grad[i] += gradOutput[i];
            }
        }
    }
}
