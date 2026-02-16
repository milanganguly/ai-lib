package milanganguly.engine;

class ExpBackward implements Operation {
    private final Tensor out;

    public ExpBackward(Tensor out) {
        this.out = out;
    }

    @Override
    public void backward(float[] gradOutput) {
        Tensor a = out.parents[0];
        if (a.requiresGrad()) {
            float[] outData = out.getData();
            float[] aGrad = a.gradRef();
            for (int i = 0; i < a.numel(); i++) {
                aGrad[i] += gradOutput[i] * outData[i];
            }
        }
    }
}
