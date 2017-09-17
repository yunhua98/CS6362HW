package cs362;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

public class NaiveBayesClassifier extends Predictor {
	private static final long serialVersionUID = 1L;

	private double[][] distribution;
	private double onesDistribution;
	private double lambda;

	public NaiveBayesClassifier(double lambda) {
		this.lambda = lambda;
	}

	public void train(List<Instance> instances) {
		int maxIndex = 0;
		for (int i = 0; i < instances.size(); ++i) {
			Instance inst = instances.get(i);
			Iterator<Integer> iter = inst.getFeatureVector().indexIterator();
			int lastIndex = 0;
			while (iter.hasNext()) {
				lastIndex = iter.next();
			}
			if (lastIndex > maxIndex) {
				maxIndex = lastIndex;
			}
		}

		distribution = new double[2][maxIndex];  // 1-index to 0-index conversion
		double[] totals = new double[2];  // total label counts for 0 and 1
		double zeros = lambda;  // number of zero labels with smoothening
		double ones = lambda;  // number of one labels with smoothening

		for (Instance inst : instances) {
			double doubleLabel = Double.parseDouble(inst.getLabel().toString());
			int label;
			if (doubleLabel < 0.5) {
				label = 0;
				++zeros;
			} else {
				label = 1;
				++ones;
			}

			FeatureVector vec = inst.getFeatureVector();
			Iterator<Integer> iter = vec.indexIterator();
			while (iter.hasNext()) {
				int index = iter.next();
				distribution[label][index - 1] += vec.get(index);  // 1-index to 0-index conversion
				totals[label] += vec.get(index);
			}
		}

		for (int i = 0; i < maxIndex; ++i) {
			distribution[0][i] += lambda;
			distribution[1][i] += lambda;
			totals[0] += lambda;
			totals[1] += lambda;
		}

		onesDistribution = ones / (zeros + ones);
		for (int i = 0; i < maxIndex; ++i) {
			distribution[0][i] /= totals[0];
			distribution[1][i] /= totals[1];
		}

	}
	
	public Label predict(Instance instance) {
		double zero = Math.log(1.0 - onesDistribution);
		double one = Math.log(onesDistribution);

		FeatureVector vec = instance.getFeatureVector();
		Iterator<Integer> iter = vec.indexIterator();

		while (iter.hasNext()) {
			int index = iter.next();
			if (index > distribution[0].length)  // 1-index to 0-index
				continue;
			zero += Math.log(distribution[0][index - 1]);
			one += Math.log(distribution[1][index - 1]);
		}

		return new ClassificationLabel((zero > one ? 0 : 1));
	}
}
