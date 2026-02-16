package milanganguly.nn.init;

import milanganguly.engine.Tensor;

public interface Initializer {
    Tensor initialize(boolean requiresGrad, int... shape);
}
