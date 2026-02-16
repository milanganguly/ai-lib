package milanganguly.engine;

class MaxRowsBackward implements Operation {
    private final Tensor a;
    private final Tensor out;
    public MaxRowsBackward(Tensor a, Tensor out) {
        this.a = a;
        this.out = out;
    }
    @Override
    public void backward(float[] gradOutput) {
        if (!a.requiresGrad()) return;
        int[] shape = a.getShape();
        int rows = shape[0];
        int cols = shape[1];
        float[] aData = a.dataRef();
        float[] aGrad = a.gradRef();
        float[] outData = out.dataRef();
        for (int i = 0; i < rows; i++) {
            float maxVal = outData[i];
            float g = gradOutput[i];
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                if (aData[idx] == maxVal) {
                    aGrad[idx] += g;
                }
            }
        }
    }
}
