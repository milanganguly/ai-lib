package milanganguly.nn.layers;

import milanganguly.nn.Module;
import milanganguly.nn.init.Initializer;
import milanganguly.nn.init.Initializers;
import milanganguly.engine.Tensor;
import milanganguly.engine.TensorOps;

import java.util.Collections;
import java.util.List;

public class MaxPool2D extends Module {

    private final int poolH;
    private final int poolW;
    private final int stride;

    public MaxPool2D(int poolH, int poolW, int stride) {
        this.poolH = poolH;
        this.poolW = poolW;
        this.stride = stride;
    }
    public MaxPool2D(int kernelSize, int stride) {
        this(kernelSize, kernelSize, stride);
    }
    @Override
    public Tensor forward(Tensor x) {
        return TensorOps.maxPool2D(x, poolH, poolW, stride);
    }
    @Override
    public List<Tensor> parameters() {
        return Collections.emptyList(); // no learnable parameters
    }
}
