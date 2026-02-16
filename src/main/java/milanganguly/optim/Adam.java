package milanganguly.optim;

import milanganguly.engine.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import milanganguly.nn.Module;

public class Adam extends Optimizer {

    private float beta1 = 0.9f;
    private float beta2 = 0.999f;
    private float eps = 1e-8f;
    private int t = 0;

    private Map<Tensor, float[]> m = new HashMap<>();
    private Map<Tensor, float[]> v = new HashMap<>();

    public Adam(Module module, float lr) {
        super(module, lr);
    }
    public Adam(List<Tensor> parameters, float lr) {
        super(parameters, lr);
    }

    @Override
    public void step() {

        t += 1;


        for (Tensor p : this.parameters) {
            if (p.requiresGrad()) {
                int pNumel = p.numel();
                float[] newGrads = new float[p.numel()];

                // lazy init m and v
                m.putIfAbsent(p, new float[pNumel]);
                v.putIfAbsent(p, new float[pNumel]);

                float[] mArr = m.get(p);
                float[] vArr = v.get(p);

                float bias1 = (float)(1 - Math.pow(beta1, t));
                float bias2 = (float)(1 - Math.pow(beta2, t));

                float[] pGrads = p.getGrad();

                for (int i = 0; i < pNumel; i++) {
                    float g = pGrads[i];

                    // Update moments
                    mArr[i] = beta1 * mArr[i] + (1 - beta1) * g;
                    vArr[i] = beta2 * vArr[i] + (1 - beta2) * (g * g);

                    // bias correction
                    float mHat = (float) (mArr[i] / bias1);
                    float vHat = (float) (vArr[i] / bias2);

                    // Parameter update
                    newGrads[i] += (float) (mHat / (Math.sqrt(vHat) + eps));
                }
                p.addScaledUpdate(newGrads, -lr);
            }
        }
    }
}
