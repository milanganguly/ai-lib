package milanganguly.engine;

class SumRowsBackward implements Operation {
    private final Tensor a;

    public SumRowsBackward(Tensor a) {
        this.a = a;
    }
    @Override
    public void backward(float[] gradOutput) {
        if (!a.requiresGrad()) return;
        int[] shape = a.getShape();
        int rows = shape[0];
        int cols = shape[1];
        float[] aGrad = a.gradRef();
        for (int i = 0; i < rows; i++) {
            float g = gradOutput[i];
            for (int j = 0; j < cols; j++) {
                aGrad[i * cols + j] += g;
            }
        }
    }
}
