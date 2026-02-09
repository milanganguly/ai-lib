package milanganguly.autograd;

import milanganguly.tensor.Tensor;

class MulBackward implements Operation {
    private final Tensor a;
    private final Tensor b;
    public MulBackward(Tensor a, Tensor b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public void backward(float[] gradOutput) {
        for (int i = 0; i < a.grad.length; i++) {
            if (a.requiresGrad) {
                a.grad[i] += gradOutput[i]*b.data[i];
            }
            if (b.requiresGrad) {
                b.grad[i] += gradOutput[i]*a.data[i];
            }
        }
    }
}
