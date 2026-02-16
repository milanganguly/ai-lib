package milanganguly.engine;

class NegBackward implements Operation {
    private final Tensor a;
    public NegBackward(Tensor a) {
        this.a = a;
    }

    @Override
    public void backward(float[] gradOutput) {
        if (a.requiresGrad()) {
            float[] aGrad = a.gradRef();
            for (int i = 0; i < a.numel(); i++) {
                aGrad[i] += -gradOutput[i];
            }
        }
    }
}
