package milanganguly.autograd;

import milanganguly.tensor.Tensor;

class AddBackward implements Operation {
    private final Tensor a;
    private final Tensor b;
    public AddBackward(Tensor a, Tensor b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public void backward(float[] gradOutput) {
        for (int i = 0; i < a.grad.length; i++) {
            if (a.requiresGrad) {
                a.grad[i] += gradOutput[i];
            }
            if (b.requiresGrad) {
                b.grad[i] += gradOutput[i];
            }
        }
    }
}
