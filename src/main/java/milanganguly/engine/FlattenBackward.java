package milanganguly.engine;

class FlattenBackward implements Operation {
    private final Tensor a;

    public FlattenBackward(Tensor a) {
        this.a = a;
    }

    @Override
    public void backward(float[] gradOutput) {
        if (a.requiresGrad()) {
            float[] aGrad = a.gradRef();
            for (int i = 0; i < a.numel(); i++) {
                aGrad[i] += gradOutput[i];
            }
        }
    }
}
