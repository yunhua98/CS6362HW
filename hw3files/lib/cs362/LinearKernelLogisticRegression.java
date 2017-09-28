package cs362;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

public class LinearKernelLogisticRegression extends KernelLogisticRegression {
	private static final long serialVersionUID = 1L;

	public LinearKernelLogisticRegression(double gradient_ascent_learning_rate, int gradient_ascent_training_iterations) {
		super(gradient_ascent_learning_rate, gradient_ascent_training_iterations);
	}

	@Override
	protected double kernelTransform(FeatureVector x, FeatureVector y) {
		double linear = 0.0;
		Iterator<Integer> iter = x.indexIterator();
		// Iterator<Integer> iterY = y.indexIterator();

		while (iter.hasNext()) {
			int index = iter.next();
			linear += x.get(index) * y.get(index);
		}

		return linear;
	}
}
