package milanganguly.autograd;

import milanganguly.tensor.Tensor;

class NegBackward implements Operation {
    private final Tensor a;
    public NegBackward(Tensor a) {
        this.a = a;
    }

    @Override
    public void backward(float[] gradOutput) {
        if (a.requiresGrad) {
            for (int i = 0; i < a.grad.length; i++) {
                a.grad[i] += -gradOutput[i];
            }
        }
    }
}
