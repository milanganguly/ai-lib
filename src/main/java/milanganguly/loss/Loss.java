package milanganguly.loss;

import milanganguly.engine.Tensor;

public abstract class Loss {

    public abstract Tensor forward(Tensor prediction, Tensor target);

}
