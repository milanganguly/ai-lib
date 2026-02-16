package milanganguly.engine;

class MatMulBackward implements Operation {
    private final Tensor a;
    private final Tensor b;
    public MatMulBackward(Tensor a, Tensor b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public void backward(float[] gradOutput) {
        int m = a.getShape()[0];
        int k = a.getShape()[1];
        int n = b.getShape()[1];
        if (a.requiresGrad()) {
            float[] bData = b.getData();
            float[] aGrad = a.gradRef();
            for (int i = 0; i < m; i++) {
                for (int p = 0; p < k; p++) {
                    float sum = 0f;
                    for (int j = 0; j < n; j++) {
                        float g = gradOutput[i * n + j]; // G[i,j]
                        float bVal = bData[p * n + j];       // B[p,j]
                        sum += g * bVal;
                    }
                    aGrad[i * k + p] += sum;
                }
            }
        }
        if (b.requiresGrad()) {
            float[] bGrad = b.gradRef();
            float[] aData = a.getData();
            for (int p = 0; p < k; p++) {
                for (int j = 0; j < n; j++) {
                    float sum = 0f;
                    for (int i = 0; i < m; i++) {
                        float g = gradOutput[i * n + j]; // G[i,j]
                        float aVal = aData[i * k + p];       // A[i,p]
                        sum += g * aVal;
                    }
                    bGrad[p * n + j] += sum;
                }
            }
        }
    }
}
