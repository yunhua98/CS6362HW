package cs362;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

public class EvenOddPredictor extends Predictor {

	public void train(List<Instance> instances) {
		// Do nothing :)
	}
	
	public Label predict(Instance instance) {
		FeatureVector features = instance.getFeatureVector();
		Iterator<Integer> iter = features.indexIterator();
		double evenSum = 0;
		double oddSum = 0;
		while (iter.hasNext()) {
			int index = iter.next();
			if (index % 2 == 0) {
				evenSum += features.get(index);
			} else {
				oddSum += features.get(index);
			}
		}
		ClassificationLabel label = new ClassificationLabel((evenSum >= oddSum ? 1 : 0));
		return label;
	}
}
