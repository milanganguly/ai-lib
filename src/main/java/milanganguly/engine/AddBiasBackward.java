package milanganguly.engine;

public class AddBiasBackward implements Operation {

    private final Tensor a;
    private final Tensor b;

    public AddBiasBackward(Tensor a, Tensor b) {
        this.a = a;
        this.b = b;
    }
    @Override
    public void backward(float[] gradOutput) {
        int[] aShape = a.getShape();
        int rows = aShape[0], cols = aShape[1];

        if (a.requiresGrad()) {
            float[] aGrad = a.gradRef();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    aGrad[i * cols + j] += gradOutput[i * cols + j];
                }
            }
        }

        if (b.requiresGrad()) {
            float[] bGrad = b.gradRef();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    bGrad[j] += gradOutput[i * cols + j];
                }
            }
        }
    }

}
