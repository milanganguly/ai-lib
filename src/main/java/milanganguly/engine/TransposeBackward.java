package milanganguly.engine;

class TransposeBackward implements Operation {
    private final Tensor a;

    public TransposeBackward(Tensor a) {
        this.a = a;
    }

    @Override
    public void backward(float[] gradOutput) {
        if (a.requiresGrad()) {
            int[] aShape = a.getShape();
            int m = aShape[0];
            int n = aShape[1];

            float[] aGrad = a.gradRef();

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    // reverse transpose
                    aGrad[i * n + j] += gradOutput[j * m + i];
                }
            }
        }
    }
}
