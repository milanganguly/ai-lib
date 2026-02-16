package milanganguly.engine;

class AddBackward implements Operation {
    private final Tensor a;
    private final Tensor b;
    public AddBackward(Tensor a, Tensor b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public void backward(float[] gradOutput) {
        float[] aGrad = a.gradRef();
        float[] bGrad = b.gradRef();
        for (int i = 0; i < a.numel(); i++) {
            if (a.requiresGrad()) {
                aGrad[i] += gradOutput[i];
            }
            if (b.requiresGrad()) {
                bGrad[i] += gradOutput[i];
            }
        }
    }
}
