package milanganguly.engine;

class LogBackward implements Operation {
    private final Tensor a;
    public LogBackward(Tensor a) {
        this.a = a;
    }
    @Override
    public void backward(float[] gradOutput) {
        if (!a.requiresGrad()) return;
        float[] aData = a.dataRef();
        float[] aGrad = a.gradRef();
        for (int i = 0; i < a.numel(); i++) {
            aGrad[i] += gradOutput[i] / aData[i];
        }
    }
}
