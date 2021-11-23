import java.util.Arrays;
import java.util.stream.IntStream;

public class Feature implements Comparable<Feature> {
    private final int[] vec;
    private double[] nVec;
    private final int[] cumSqVec;
    private final double invNorm;
    private boolean visited;

    public Feature(int[] vec) {
        double _invNorm = Utils.invL2Norm(vec);
        this.vec = vec;
        this.invNorm = _invNorm;

        this.nVec = new double[vec.length];
        int[] sqVec = new int[vec.length];
        IntStream.range(0, vec.length).forEach(i -> {
            nVec[i] = vec[i] * _invNorm;
            sqVec[i] = vec[i] * vec[i];
        });

        Arrays.parallelPrefix(sqVec, Integer::sum);
        this.cumSqVec = sqVec;
        this.visited = false;
    }

    public Feature(int n) {
        this.vec = new int[n];
        this.invNorm = 0;
        this.nVec = new double[n];
        this.cumSqVec = new int[n];
        this.visited = false;
    }

    public int[] getVec() {
        return vec;
    }

    public double[] getnVec() {
        return nVec;
    }

    public int[] getCumSqVec() {
        return cumSqVec;
    }

    public double getInvNorm() {
        return invNorm;
    }

    public boolean isVisited() {
        return visited;
    }

    public void setnVec(double[] nVec) {
        this.nVec = nVec;
    }

    public void setVisited(boolean visited) {
        this.visited = visited;
    }

    @Override
    public int compareTo(Feature o) {
        return this.invNorm < o.getInvNorm() ? 1 : -1;
    }
}
