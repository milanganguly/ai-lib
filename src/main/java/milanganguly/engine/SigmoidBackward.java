package milanganguly.engine;

class SigmoidBackward implements Operation {
    private final Tensor out;

    public SigmoidBackward(Tensor out) {
        this.out = out;
    }

    @Override
    public void backward(float[] gradOutput) {
        Tensor a = out.parents[0];
        float[] aGrad = a.gradRef();
        if (a.requiresGrad()) {
            for (int i = 0; i < a.numel(); i++) {
                float y = out.getData()[i];
                aGrad[i] += gradOutput[i] * y * (1.0f - y);
            }
        }
    }
}

