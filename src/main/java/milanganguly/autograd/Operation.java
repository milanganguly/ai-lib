package milanganguly.autograd;

import milanganguly.tensor.Tensor;

public interface Operation {
    void backward(float[] gradOutput);
}
