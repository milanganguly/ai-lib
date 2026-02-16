package milanganguly.optim;

import milanganguly.engine.Tensor;

import java.util.List;
import milanganguly.nn.Module;

public abstract class Optimizer {
    protected List<Tensor> parameters;
    protected float lr;

    public Optimizer(Module module, float lr) {
        this.parameters = module.parameters();
        this.lr = lr;
    }
    public Optimizer(List<Tensor> parameters, float lr) {
        this.parameters = parameters;
        this.lr = lr;
    }
    public abstract void step();
    public void zeroGrad() {
        for (Tensor p : parameters) {
            p.zeroGrad();
        }
    }
}

