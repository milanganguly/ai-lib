package milanganguly.autograd;

import milanganguly.tensor.Tensor;

class MatMulBackward implements Operation {
    private final Tensor a;
    private final Tensor b;
    public MatMulBackward(Tensor a, Tensor b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public void backward(float[] gradOutput) {
        int m = a.shape[0];
        int k = a.shape[1];
        int n = b.shape[1];
        if (a.requiresGrad) {
            for (int i = 0; i < m; i++) {
                for (int p = 0; p < k; p++) {
                    float sum = 0f;
                    for (int j = 0; j < n; j++) {
                        float g = gradOutput[i * n + j]; // G[i,j]
                        float bVal = b.data[p * n + j];       // B[p,j]
                        sum += g * bVal;
                    }
                    a.grad[i * k + p] += sum;
                }
            }
        }
        if (b.requiresGrad) {
            for (int p = 0; p < k; p++) {
                for (int j = 0; j < n; j++) {
                    float sum = 0f;
                    for (int i = 0; i < m; i++) {
                        float g = gradOutput[i * n + j]; // G[i,j]
                        float aVal = a.data[i * k + p];       // A[i,p]
                        sum += g * aVal;
                    }
                    b.grad[p * n + j] += sum;
                }
            }
        }
    }
}
