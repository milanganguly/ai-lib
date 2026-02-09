package milanganguly.autograd;

import milanganguly.tensor.Tensor;

class SigmoidBackward implements Operation {
    private final Tensor out;

    public SigmoidBackward(Tensor out) {
        this.out = out;
    }

    @Override
    public void backward(float[] gradOutput) {
        Tensor a = out.parents[0];
        if (a.requiresGrad) {
            for (int i = 0; i < a.numel(); i++) {
                float y = out.data[i];
                a.grad[i] += gradOutput[i] * y * (1.0f - y);
            }
        }
    }
}

