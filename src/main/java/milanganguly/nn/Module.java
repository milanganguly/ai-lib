package milanganguly.nn;

import milanganguly.engine.Tensor;
import java.util.*;

public abstract class Module {

    private final List<Tensor> parameters = new ArrayList<>();
    private final List<Module> children = new ArrayList<>();

    public abstract Tensor forward(Tensor x);

    protected void registerParameter(Tensor t) {
        if (t != null) {
            parameters.add(t);
        }
    }
    protected void registerModule(Module m) {
        if (m != null) {
            children.add(m);
        }
    }
    public List<Tensor> parameters() {
        List<Tensor> all = new ArrayList<>(parameters);
        for (Module child : children) {
            all.addAll(child.parameters());
        }
        return all;
    }
    public void zeroGrad() {
        for (Tensor p : parameters()) {
            p.zeroGrad();
        }
    }
}
