package cs362;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

public class PerceptronClassifier extends Predictor {
	private static final long serialVersionUID = 1L;

	private double online_learning_rate;
	private int online_training_iterations;
	private double[] weights;

	public PerceptronClassifier(double online_learning_rate, int online_training_iterations) {
		this.online_learning_rate = online_learning_rate;
		this.online_training_iterations = online_training_iterations;
	}

	public void train(List<Instance> instances) {
		int maxIndex = 0;
		for (int i = 0; i < instances.size(); ++i) {
			Instance inst = instances.get(i);
			FeatureVector vec = inst.getFeatureVector();
			if (vec.size() > maxIndex) {
				maxIndex = vec.size();
			}
		}

		weights = new double[maxIndex];  // 1-index to 0-index conversion

		for (int k = 0; k < online_training_iterations; ++k) {
			for (Instance inst : instances) {
				double observedLabel = Double.parseDouble(inst.getLabel().toString());
				if (observedLabel == 0.0) {
					observedLabel = -1.0;
				}
				FeatureVector features = inst.getFeatureVector();
				Iterator<Integer> iter = features.indexIterator();  // 1-indexed
				double dotProduct = 0.0;
				while (iter.hasNext()) {
					int index = iter.next();
					dotProduct += weights[index - 1] * features.get(index);
				}

				double predictedLabel = (dotProduct >= 0 ? 1 : -1);

				if (predictedLabel != observedLabel) {
					iter = features.indexIterator();
					while (iter.hasNext()) {
						int index = iter.next();
						weights[index - 1] += online_learning_rate * observedLabel * features.get(index);
					}
				}
			}
		}
	}
	
	public Label predict(Instance instance) {
		double dotProduct = 0.0;
		FeatureVector features = instance.getFeatureVector();
		Iterator<Integer> iter = features.indexIterator();
		while (iter.hasNext()) {
			int index = iter.next();
			dotProduct += weights[index - 1] * features.get(index);
		}
		return new ClassificationLabel((dotProduct >= 0 ? 1 : 0));
	}
}
