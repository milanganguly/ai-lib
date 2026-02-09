package milanganguly.autograd;

import milanganguly.tensor.Tensor;

import java.util.*;

public class Autograd {
    public static void backward(Tensor root) {
        Arrays.fill(root.grad, 1f);
        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        buildTopo(root, topo, visited);
        Collections.reverse(topo);
        for (Tensor t : topo) {
            if (t.backwardsOp!=null) {
                t.backwardsOp.backward(t.grad);
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
    public static Operation expBackward(Tensor a) {
        return new ExpBackward(a);
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
}
