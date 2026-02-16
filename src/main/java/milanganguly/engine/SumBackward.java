package milanganguly.engine;

class SumBackward implements Operation {
    private final Tensor a;
    public SumBackward(Tensor a) {
        this.a = a;
    }

    @Override
    public void backward(float[] gradOutput) {
        if (a.requiresGrad()) {
            float[] aGrad = a.gradRef();
            for (int i = 0; i < a.getGrad().length; i++) {
                aGrad[i] += gradOutput[0];
            }
        }
    }
}
