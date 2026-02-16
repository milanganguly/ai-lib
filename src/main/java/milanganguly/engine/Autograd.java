package milanganguly.engine;

import java.util.*;

public class Autograd {
    public static void backward(Tensor root) {
        root.fillGrad(1);
        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        buildTopo(root, topo, visited);
        Collections.reverse(topo);
        for (Tensor t : topo) {
            if (t.backwardsOp!=null) {
                t.backwardsOp.backward(t.getGrad());
            }
        }
    }
    public static void buildTopo(Tensor t, List<Tensor> topo, Set<Tensor> visited) {
        if (!visited.contains(t)) {
            visited.add(t);
            if (t.parents!=null) {
                for (Tensor p : t.parents) {
                    buildTopo(p, topo, visited);
                }
            }
            topo.add(t);
        }
    }
    public static Operation matMulBackward(Tensor a, Tensor b) {
        return new MatMulBackward(a, b);
    }
    public static Operation addBackward(Tensor a, Tensor b) {
        return new AddBackward(a, b);
    }
    public static Operation mulBackward(Tensor a, Tensor b) {
        return new MulBackward(a, b);
    }
    public static Operation negBackward(Tensor a) {
        return new NegBackward(a);
    }
    public static Operation ReLUBackward(Tensor a) {
        return new ReLUBackward(a);
    }
    public static Operation sumBackward(Tensor a) {
        return new SumBackward(a);
    }
    public static Operation expBackward(Tensor out) {
        return new ExpBackward(out);
    }
    public static Operation sigmoidBackward(Tensor out) {
        return new SigmoidBackward(out);
    }
    public static Operation reshapeBackward(Tensor a) {
        return new ReshapeBackward(a);
    }
    public static Operation transposeBackward(Tensor a) {
        return new TransposeBackward(a);
    }
    public static Operation addBiasBackward(Tensor a, Tensor b) {
        return new AddBiasBackward(a, b);
    }
    public static Operation im2colBackward(Tensor input, int kernelH, int kernelW, int stride, int padding, int H_out, int W_out) {
        return new Im2colBackward(input, kernelH, kernelW, stride, padding, H_out, W_out);
    }
    public static Operation temp(Tensor a) {
        return new FlattenBackward(a);
    }
    public static Operation absBackward(Tensor a) {
        return new AbsBackward(a);
    }
    public static Operation sumRowsBackward(Tensor a) {
        return new SumRowsBackward(a);
    }
    public static Operation maxRowsBackward(Tensor a, Tensor out) {
        return new MaxRowsBackward(a, out);
    }
    public static Operation logBackward(Tensor a) {
        return new LogBackward(a);
    }
    public static Operation subtractRowBackward(Tensor a, Tensor rowVec) {
        return new SubtractRowBackward(a, rowVec);
    }
    public static Operation maxPool2DBackward(Tensor input, int poolH, int poolW, int stride, int H_out, int W_out) {
        return new MaxPool2DBackward(input, poolH, poolW, stride, H_out, W_out);
    }
}
