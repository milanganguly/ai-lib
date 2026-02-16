package milanganguly.nn.layers;

import milanganguly.nn.Module;
import milanganguly.nn.init.Initializer;
import milanganguly.nn.init.Initializers;
import milanganguly.engine.Tensor;
import milanganguly.engine.TensorOps;

public class Conv2D extends Module {

    // outC, inC, kH, kW
    private final Tensor weight;
    private final Tensor bias;

    private final int stride;
    private final int padding;

    public Conv2D(int inChannels, int outChannels, int kernelH, int kernelW, int stride, int padding, Initializer weightInit, Initializer biasInit) {
        this.weight = weightInit.initialize(true, outChannels, inChannels, kernelH, kernelW);
        this.bias   = biasInit.initialize(true, outChannels);
        this.stride = stride;
        this.padding = padding;
        registerParameter(this.weight);
        registerParameter(this.bias);
    }
    public Conv2D(int inChannels, int outChannels, int kernelH, int kernelW) {
        this(inChannels, outChannels, kernelH, kernelW,
                1, 0,
                Initializers.xavier(), Initializers.zeros());
    }
    @Override
    public Tensor forward(Tensor x) {
        // x: [C, H, W]
        // im2col: [OH*OW, C*kH*kW]
        Tensor xCol = TensorOps.im2col(
                x,
                this.weight.getShape()[2],
                this.weight.getShape()[3],
                stride,
                padding
        );

        // wCol: [outC, C*kH*kW]
        Tensor wCol = TensorOps.reshape(
                this.weight,
                new int[]{
                        this.weight.getShape()[0],
                        this.weight.getShape()[1] * this.weight.getShape()[2] * this.weight.getShape()[3]
                }
        );

        // yCol: [OH*OW, outC]  (so addBias works with bias=[outC])
        Tensor yCol = TensorOps.matmul(xCol, TensorOps.transpose(wCol));
        Tensor yColBiased = TensorOps.addBias(yCol, this.bias);

        // reshape to [outC, OH, OW]
        int H = x.getShape()[1];
        int W = x.getShape()[2];
        int OH = (H + 2 * padding - weight.getShape()[2]) / stride + 1;
        int OW = (W + 2 * padding - weight.getShape()[3]) / stride + 1;

        Tensor y = TensorOps.transpose(yColBiased); // [outC, OH*OW]
        return TensorOps.reshape(y, new int[]{weight.getShape()[0], OH, OW});
    }

}
