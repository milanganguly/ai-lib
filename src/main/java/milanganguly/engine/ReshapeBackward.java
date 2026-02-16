package milanganguly.engine;

class ReshapeBackward implements Operation {
    private final Tensor a;

    public ReshapeBackward(Tensor a) {
        this.a = a;
    }

    @Override
    public void backward(float[] gradOutput) {
        float[] aGrad = a.gradRef();
        if (a.requiresGrad()) {
            for (int i = 0; i < a.numel(); i++) {
                aGrad[i] += gradOutput[i];
            }
        }
    }
}
