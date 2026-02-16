package milanganguly.engine;

class AbsBackward implements Operation {
    private final Tensor a;

    public AbsBackward(Tensor a) {
        this.a = a;
    }

    @Override
    public void backward(float[] gradOutput) {
        if (!a.requiresGrad()) return;

        float[] aGrad = a.gradRef();
        float[] aData = a.dataRef();

        for (int i = 0; i < a.numel(); i++) {
            float sign;
            if (aData[i] > 0f) {
                sign = 1f;
            } else if (aData[i] < 0f) {
                sign = -1f;
            } else {
                sign = 0f;
            }

            aGrad[i] += sign * gradOutput[i];
        }
    }
}
