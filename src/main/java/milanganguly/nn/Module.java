package milanganguly.nn;

import milanganguly.engine.Tensor;
import java.util.*;
import java.io.*;

public abstract class Module {

    private final List<Tensor> parameters = new ArrayList<>();
    private final List<Module> children = new ArrayList<>();

    public abstract Tensor forward(Tensor x);

    protected void registerParameter(Tensor t) {
        if (t != null) {
            parameters.add(t);
        }
    }
    protected void registerModule(Module m) {
        if (m != null) {
            children.add(m);
        }
    }
    public List<Tensor> parameters() {
        List<Tensor> all = new ArrayList<>(parameters);
        for (Module child : children) {
            all.addAll(child.parameters());
        }
        return all;
    }
    public void zeroGrad() {
        for (Tensor p : parameters()) {
            p.zeroGrad();
        }
    }
    public void save(String path) {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
            out.writeInt(1);
            List<Tensor> params = this.parameters();
            out.writeInt(params.size());
            for (Tensor p : params) {
                int[] shape = p.getShape();
                out.writeInt(shape.length);
                for (int dim : shape) {out.writeInt(dim);}
                float[] data = p.getData();
                out.writeInt(data.length);
                for (float v : data) {
                    out.writeFloat(v);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Error saving model", e);
        }
    }
    public void load(String path) {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            int version = in.readInt();
            if (version != 1) {
                throw new RuntimeException("Unsupported model version: " + version);
            }
            List<Tensor> params = this.parameters();
            int paramCount = in.readInt();
            if (paramCount != params.size()) {
                throw new RuntimeException(
                        "Parameter count mismatch: file has " +
                                paramCount + " but model has " + params.size()
                );
            }
            for (Tensor p : params) {
                int shapeLen = in.readInt();
                int[] fileShape = new int[shapeLen];
                for (int i = 0; i < shapeLen; i++) {
                    fileShape[i] = in.readInt();
                }
                int[] modelShape = p.getShape();
                if (!Arrays.equals(fileShape, modelShape)) {
                    throw new RuntimeException("Shape mismatch: expected " + Arrays.toString(modelShape) + " but found " + Arrays.toString(fileShape));
                }
                int length = in.readInt();
                if (length != p.numel()) {
                    throw new RuntimeException("Data length mismatch: expected " + p.numel() + " but found " + length);
                }
                float[] loaded = new float[length];
                for (int i = 0; i < length; i++) {
                    loaded[i] = in.readFloat();
                }
                p.copyFrom(loaded);
            }

        } catch (IOException e) {
            throw new RuntimeException("Error loading model", e);
        }
    }


}
