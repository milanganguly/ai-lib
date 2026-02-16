package milanganguly.nn.init;


import milanganguly.engine.Tensor;

class Zeros implements Initializer {

    @Override
    public Tensor initialize(boolean requiresGrad, int... shape) {
        int size = 1;
        for (int s : shape) {
            size*=s;
        }
        return new Tensor(new float[size], requiresGrad, shape);
    }
}
