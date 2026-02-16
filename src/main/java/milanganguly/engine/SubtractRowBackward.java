package milanganguly.engine;

class SubtractRowBackward implements Operation {
    private final Tensor a;
    private final Tensor rowVec;
    public SubtractRowBackward(Tensor a, Tensor rowVec) {
        this.a = a;
        this.rowVec = rowVec;
    }
    @Override
    public void backward(float[] gradOutput) {
        int[] shape = a.getShape();
        int rows = shape[0];
        int cols = shape[1];
        if (a.requiresGrad()) {
            float[] aGrad = a.gradRef();

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    aGrad[i * cols + j] += gradOutput[i * cols + j];
                }
            }
        }
        if (rowVec.requiresGrad()) {
            float[] rowGrad = rowVec.gradRef();
            for (int i = 0; i < rows; i++) {
                float sum = 0f;
                for (int j = 0; j < cols; j++) {
                    sum += gradOutput[i * cols + j];
                }
                rowGrad[i] -= sum; // its rowVec subtract, hence negate gradient
            }
        }
    }
}
