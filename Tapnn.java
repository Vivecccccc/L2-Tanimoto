import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class Tapnn {
    private Map<String, Feature> map;
    private double[] maxF;
    private Map<String, Pair> pfMap;
    private Map<Integer, List<PrefInfo>> idMap;
    private Map<Integer, Integer> stepStone;
    private Map<String, Double> accumulator;
    public Map<String, List<String>> resMap;

    private double sim = 0.95;
    private double invCosThres = 0.5 * 1.0526316f + 0.5;
    private float cosThres = 0.974358974;
    private double alpha = 0.5 * (2.05263158 + 0.461840231);

    public Tapnn(Map<String, int[]> map) {
        dataPrep(map);
        candidateGen();
        candidateValid();
    }

    private void dataPrep(Map<String, int[]> _map) {
        Map<String, Feature> fMap = _map.entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, e -> new Feature(e.getValue())));
        Map<String, Feature> sFMap = fMap.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue())
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
        double[] maxFeature = Utils.genMaxFeature(sFMap);

        this.map = sFMap;
        this.maxF = maxFeature;

        this.pfMap = new LinkedHashMap<>();
        this.idMap = IntStream.range(0, this.maxF.length).boxed().collect(Collectors.toMap(Function.identity(), i -> new ArrayList<>()));

        this.stepStone = IntStream.range(0, this.maxF.length).boxed().collect(Collectors.toMap(Function.identity(), i -> 0));
        this.accumulator = this.map.entrySet().stream().collect(Collectors.toMap(Map.Entry::getKey, e -> 0.0));

        this.resMap = new LinkedHashMap<>();
    }

    private void candidateGen() {
        for (Map.Entry<String, Feature> entry : this.map.entrySet()) {
            String sign = entry.getKey();
            Feature feat = entry.getValue();

            double[] nVec = feat.getnVec();

            boolean flagPref = false;
            int j = 0;
            while (j < nVec.length && !flagPref) {
                double jVal = nVec[j];
                if (jVal > 0) {
                    double invPScore = Utils.calcInvPref(this.maxF, j, feat);
                    if (invPScore <= this.invCosThres) {
                        flagPref = true;
                        pfMap.put(sign, new Pair(j, Utils.calcInvPref(this.maxF, j - 1, feat)));
                        break;
                    }
                }
                j += 1;
            }
            while (j < nVec.length && flagPref) {
                double jVal = nVec[j];
                if (jVal > 0) {
                    double jInvPrefVal = Utils.invSqrt(feat.getCumSqVec()[j - 1]);
                    PrefInfo info = new PrefInfo(sign, jVal, jInvPrefVal);
                    idMap.computeIfAbsent(j, k -> new ArrayList<>()).add(info);
                }
                j += 1;
            }
        }
    }

    private void candidateValid() {
        String prevSign = this.map.entrySet().iterator().next().getKey();
        String initSign = prevSign;
        for (Map.Entry<String, Feature> entry : this.map.entrySet()) {
            String sign = entry.getKey();
            Feature feat = entry.getValue();
            int p = pfMap.get(sign).getIdx();
            double[] nVec = feat.getnVec();
            double invNorm = feat.getInvNorm();
            double invLenThres = alpha * invNorm;
            this.accumulator = this.map.entrySet().stream().collect(Collectors.toMap(Map.Entry::getKey, e -> 0.0));

            for (int j = nVec.length - 1; j >= 0; j--) {
                int step = stepStone.get(j);
                int _len = idMap.get(j).size();
                for (int k = step; k < _len; k++) {
                    List<PrefInfo> ctx = idMap.get(j);
                    String ctxSign = ctx.get(k).getSign();
                    if (feat.isVisited()) {
                        accumulator.put(ctxSign, 0.0);
                        continue;
                    }

                    if (feat.getInvNorm() >= invLenThres) {
                        stepStone.computeIfPresent(j, (key, val) -> val + 1);
                    }
                    else if (this.map.get(ctxSign).getInvNorm() < this.map.get(sign).getInvNorm() || ctxSign.equals(sign)) {
                        break;
                    }
                    else {
                        double invPScore = Utils.calcInvPref(maxF, j, feat);
                        int normQSq = this.map.get(sign).getCumSqVec()[this.map.get(sign).getCumSqVec().length - 1];
                        int norm0Sq = this.map.get(initSign).getCumSqVec()[this.map.get(initSign).getCumSqVec().length - 1];
                        int normPSq = this.map.get(prevSign).getCumSqVec()[this.map.get(prevSign).getCumSqVec().length - 1];
                        if (accumulator.get(ctxSign) > 0 ||
                                (invPScore <= invCosThres && Utils.invSqrt(feat.getCumSqVec()[p - 1]) <= Utils.eq8(norm0Sq, normPSq, normQSq, invPScore))) {
                            double ctxA = accumulator.get(ctxSign);
                            double delta = nVec[j] * this.map.get(ctxSign).getnVec()[j];
                            double factor = Utils.invSqrt(feat.getCumSqVec()[j - 1] * this.map.get(ctxSign).getCumSqVec()[j - 1]);
                            accumulator.put(ctxSign, ctxA + delta);
                            if (ctxA + delta + 1 / factor < cosThres) {
                                accumulator.put(ctxSign, 0.0);
                            }
                        }
                    }
                }
            }
            Map<String, Double> effAccumulator = accumulator.entrySet().stream()
                    .filter(e -> e.getValue() > 0)
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
            for (Iterator<Map.Entry<String, Double>> iterator = effAccumulator.entrySet().iterator(); iterator.hasNext(); ) {
                Map.Entry<String, Double> effEntry = iterator.next();
                String ctxSign = effEntry.getKey();
                float ctxAcc = effEntry.getValue().floatValue();
                int ctxPPos = pfMap.get(ctxSign).getIdx();
                float ctxPScore = 1 / (float) pfMap.get(ctxSign).getPrfSc();
                double[] ctxNVec = feat.getnVec();
                boolean preEscape = false;

                if (ctxAcc + ctxPScore < cosThres) {
                    iterator.remove();
                    accumulator.put(ctxSign, 0.0);
                    continue;
                }
                float s = ctxAcc + ctxPScore;
                double beta = (s / cosThres) + 1 / (Utils.invSqrt((s / cosThres) * (s / cosThres) - 1));
                if (this.map.get(ctxSign).getInvNorm() * (1 / beta) > feat.getInvNorm()) {
                    iterator.remove();
                    accumulator.put(ctxSign, 0.0);
                    continue;
                }
                int start = Math.min(ctxPPos, nVec.length) - 1;
                for (int j = start; j >= 0; j--) {
                    ctxAcc = effAccumulator.get(ctxSign).floatValue();
                    effAccumulator.put(ctxSign, ctxAcc + ctxNVec[j] * nVec[j]);
                    double delta = j == 0 ? 0 : Utils.invSqrt(feat.getCumSqVec()[j - 1] * this.map.get(ctxSign).getCumSqVec()[j - 1]);
                    if (effAccumulator.get(ctxSign) + (1 / delta) < cosThres) {
                        iterator.remove();
                        accumulator.put(ctxSign, 0.0);
                        preEscape = true;
                        break;
                    }
                }
                if (preEscape) {
                    continue;
                }
                double score = Utils.calcSim(effAccumulator.get(ctxSign),
                        feat.getCumSqVec()[feat.getCumSqVec().length - 1],
                        this.map.get(ctxSign).getCumSqVec()[this.map.get(ctxSign).getCumSqVec().length - 1],
                        feat.getInvNorm(), this.map.get(ctxSign).getInvNorm());
                if (score > 0.95) {
                    resMap.computeIfAbsent(ctxSign, k -> new ArrayList<>()).add(sign);
                    feat.setVisited(true);
                }
            }
            prevSign = sign;
        }
        System.out.println(resMap);
    }
}

class Pair {
    private int idx;
    private double prfSc;

    public Pair(int idx, double prfSc) {
        this.idx = idx;
        this.prfSc = prfSc;
    }

    public double getPrfSc() {
        return prfSc;
    }

    public int getIdx() {
        return idx;
    }

    public void setPrfSc(double prfSc) {
        this.prfSc = prfSc;
    }

    public void setIdx(int idx) {
        this.idx = idx;
    }
}

class PrefInfo {
    private String sign;
    private double jVal;
    private double jInvPrefVal;

    public PrefInfo(String sign, double jVal, double jInvPrefVal) {
        this.sign = sign;
        this.jVal = jVal;
        this.jInvPrefVal = jInvPrefVal;
    }

    public double getjInvPrefVal() {
        return jInvPrefVal;
    }

    public double getjVal() {
        return jVal;
    }

    public String getSign() {
        return sign;
    }

    public void setjInvPrefVal(double jInvPrefVal) {
        this.jInvPrefVal = jInvPrefVal;
    }

    public void setjVal(double jVal) {
        this.jVal = jVal;
    }

    public void setSign(String sign) {
        this.sign = sign;
    }
}
