package milanganguly.engine;

class MulBackward implements Operation {
    private final Tensor a;
    private final Tensor b;
    public MulBackward(Tensor a, Tensor b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public void backward(float[] gradOutput) {
        float[] aGrad = a.gradRef();
        float[] bGrad = b.gradRef();
        float[] aData = a.getData();
        float[] bData = b.getData();
        for (int i = 0; i < a.numel(); i++) {
            if (a.requiresGrad()) {
                aGrad[i] += gradOutput[i]* bData[i];
            }
            if (b.requiresGrad()) {
                bGrad[i] += gradOutput[i]* aData[i];
            }
        }
    }
}
