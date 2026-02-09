package milanganguly.autograd;

public interface Operation {
    void backward(float[] gradOutput);
}
