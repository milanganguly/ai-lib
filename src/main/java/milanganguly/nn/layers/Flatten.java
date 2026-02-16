package milanganguly.nn.layers;

import milanganguly.nn.Module;
import milanganguly.engine.Tensor;
import milanganguly.engine.TensorOps;

public class Flatten extends Module {
    @Override
    public Tensor forward(Tensor x) {
        return TensorOps.flatten(x);
    }
}
