package milanganguly.nn.layers;

import milanganguly.nn.Module;
import milanganguly.nn.init.Initializer;
import milanganguly.nn.init.Initializers;
import milanganguly.engine.Tensor;
import milanganguly.engine.TensorOps;

public class Linear extends Module {

    private final Tensor weight;
    private final Tensor bias;

    public Linear(int in, int out, Initializer weight, Initializer bias) {
        this.weight = weight.initialize(true, in, out);
        this.bias = bias.initialize(true, out);

        registerParameter(this.weight);
        registerParameter(this.bias);
    }
    public Linear(int inFeatures, int outFeatures) {
        this(inFeatures,
                outFeatures,
                Initializers.xavier(),
                Initializers.zeros());
    }
    @Override
    public Tensor forward(Tensor x) {
        // if input is 1D
        if (x.getShape().length == 1) {
            x = TensorOps.reshape(x, new int[]{1, x.numel()});
        }

        Tensor out = TensorOps.addBias(TensorOps.matmul(x, this.weight), bias);
        return out;
    }
}
