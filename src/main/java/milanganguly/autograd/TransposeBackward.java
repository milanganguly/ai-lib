package milanganguly.autograd;

import milanganguly.tensor.Tensor;

class TransposeBackward implements Operation {
    private final Tensor a;

    public TransposeBackward(Tensor a) {
        this.a = a;
    }

    @Override
    public void backward(float[] gradOutput) {
        if (a.requiresGrad) {
            int m = a.shape[0];
            int n = a.shape[1];

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    // reverse transpose
                    a.grad[i * n + j] += gradOutput[j * m + i];
                }
            }
        }
    }
}
