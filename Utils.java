import java.util.Arrays;
import java.util.Map;

public class Utils {
    public static double invL2Norm(int[] v) {
        float square = Arrays.stream(v).reduce(0, (acc, cur) -> acc += cur * cur);
        return invSqrt(square);
    }

    public static double[] genMaxFeature(Map<String, Feature> map) {
        Feature _feature = map.values().stream().reduce(new Feature(10000), (acc, cur) -> {
            double[] preVec = acc.getnVec();
            double[] curVec = cur.getnVec();
            preVec = growth(preVec, curVec.length);
            for (int i = 0; i < curVec.length; i++) {
                double preVal = preVec[i];
                double curVal = curVec[i];
                preVec[i] = Math.max(preVal, curVal);
            }
            acc.setnVec(preVec);
            return acc;
        });
        return _feature.getnVec();
    }

    public static double calcInvPref(double[] maxF, int pos, Feature feature) {
//        double[] prefVec = IntStream.range(0, pos + 1).asDoubleStream().map(i -> feature.getnVec()[(int) i]).toArray();
//        double[] prefVec = Arrays.copyOf(feature.getnVec(), pos + 1);
        double invNorm = feature.getInvNorm();
        double prod = prefInvProd(feature.getnVec(), maxF, pos);
        return Math.max(invNorm, prod);
    }

    public static double eq8(int initSq, int prevSq, int currSq, double invPScore) {
        double coef = 0.487179487f * 0.487179487f;
        double x = (initSq + currSq) * (initSq + currSq) * (invPScore * invPScore) / (prevSq * prevSq);
        return coef * x;
    }

    public static double calcSim(double acc, int normQSq, int normCtxSq, double invNormQ, double invNormCtx) {
        return acc / ((normQSq + normCtxSq) * (invNormQ * invNormCtx) - acc);
    }

    private static double prefInvProd(double[] prefVec, double[] maxF, int pos) {
        int i = 0;
        double res = 0f;
        while (i <= pos) {
            res += prefVec[i] * maxF[i];
            i += 1;
        }
        return 1 / res;
    }

    private static double[] growth(double[] vec, int n) {
        if (vec.length >= n) {
            return vec;
        }
        double[] extVec = Arrays.copyOf(vec, vec.length * 2);
        return growth(extVec, n);
    }

    public static double invSqrt(float x) {
        float xhalf = 0.5f * x;
        int i = Float.floatToIntBits(x);
        i = 0x5f375a86 - (i >> 1);
        x = Float.intBitsToFloat(i);
        x = x * (1.5f - (xhalf * x * x));
        return x;
//        return 1 / Math.sqrt(x);
    }
}
