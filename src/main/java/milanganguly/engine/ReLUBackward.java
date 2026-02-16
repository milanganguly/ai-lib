package milanganguly.engine;

class ReLUBackward implements Operation {
    private final Tensor a;
    public ReLUBackward(Tensor a) {
        this.a = a;
    }

    @Override
    public void backward(float[] gradOutput) {
        if (a.requiresGrad()) {
            float[] aGrad = a.gradRef();
            for (int i = 0; i < a.numel(); i++) {
                if (a.getData()[i] > 0) {
                    aGrad[i] += gradOutput[i];
                }
            }
        }
    }
}
