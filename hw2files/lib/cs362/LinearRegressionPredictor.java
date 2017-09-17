package cs362;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;
import org.apache.commons.math3.linear.*;

public class LinearRegressionPredictor extends Predictor {
	private static final long serialVersionUID = 1L;

	private RealVector weights = null;

	public void train(List<Instance> instances) {
		ArrayRealVector observedOutput = new ArrayRealVector(instances.size());
		int maxIndex = 0;
		for (int i = 0; i < instances.size(); ++i) {
			Instance inst = instances.get(i);
			observedOutput.setEntry(i, Double.parseDouble(inst.getLabel().toString()));
			Iterator<Integer> iter = inst.getFeatureVector().indexIterator();
			int lastIndex = 0;
			while (iter.hasNext()) {
				lastIndex = iter.next();
			}
			if (lastIndex > maxIndex) {
				maxIndex = lastIndex;
			}
		}

		Array2DRowRealMatrix features = new Array2DRowRealMatrix(instances.size(), maxIndex + 1);
		for (int i = 0; i < instances.size(); ++i) {
			FeatureVector vec = instances.get(i).getFeatureVector();
			Iterator<Integer> iter = vec.indexIterator();
			features.setEntry(i, 0, 1);
			while (iter.hasNext()) {
				int index = iter.next();
				features.setEntry(i, index, vec.get(index));
			}
		}

		RealMatrix featuresTransposed = features.transpose();
		LUDecomposition lu = new LUDecomposition(featuresTransposed.multiply(features));
		weights = lu.getSolver().getInverse().multiply(featuresTransposed).operate(observedOutput);
	}
	
	public Label predict(Instance instance) {
		FeatureVector featureVec = instance.getFeatureVector();
		Iterator<Integer> iter = featureVec.indexIterator();
		double regressionValue = weights.getEntry(0);
		while (iter.hasNext()) {
			int index = iter.next();
			regressionValue += weights.getEntry(index) * featureVec.get(index);
		}

		return new RegressionLabel(regressionValue);
	}
}
