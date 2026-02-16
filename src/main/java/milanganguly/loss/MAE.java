package milanganguly.loss;

import milanganguly.engine.Tensor;
import milanganguly.engine.TensorOps;

public class MAE extends Loss {
    @Override
    public Tensor forward(Tensor prediction, Tensor target) {
        return TensorOps.mae(prediction, target);
    }
}
