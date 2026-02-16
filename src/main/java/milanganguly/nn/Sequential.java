package milanganguly.nn;

import milanganguly.engine.Tensor;

import java.util.ArrayList;

public class Sequential extends Module {
    public final ArrayList<Module> layers = new ArrayList<>();
    public Sequential (Module... modules) {
        for (Module module : modules) {
            registerModule(module);
            layers.add(module);
        }
    }
    @Override
    public Tensor forward(Tensor x) {
        for (Module module : layers) {
            x = module.forward(x);
        }
        return x;
    }
}
