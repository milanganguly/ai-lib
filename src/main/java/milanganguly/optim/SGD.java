package milanganguly.optim;

import milanganguly.engine.Tensor;
import milanganguly.nn.Module;

import java.util.List;

public class SGD extends Optimizer {

    public SGD(Module module, float lr) {
        super(module, lr);
    }
    public SGD(List<Tensor> parameters, float lr) {
        super(parameters, lr);
    }

    @Override
    public void step() {
        for (Tensor p : this.parameters) {
            if (p.requiresGrad()) {
                p.addScaledGrad(-lr);
            }
        }
    }
}
