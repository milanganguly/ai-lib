package milanganguly.loss;

import milanganguly.engine.Tensor;
import milanganguly.engine.TensorOps;

public class CrossEntropy extends Loss {
    @Override
    public Tensor forward(Tensor prediction, Tensor target) {
        return TensorOps.crossEntropy(prediction, target);
    }
}
