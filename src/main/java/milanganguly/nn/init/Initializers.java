package milanganguly.nn.init;

public class Initializers {
    public static Initializer zeros() {
        return new Zeros();
    }
    public static Initializer xavier() {
        return new Xavier();
    }
}
