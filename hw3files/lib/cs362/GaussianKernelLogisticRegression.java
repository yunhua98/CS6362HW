package cs362;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

public class GaussianKernelLogisticRegression extends KernelLogisticRegression {
	private static final long serialVersionUID = 1L;

	private double gaussian_kernel_sigma;

	public GaussianKernelLogisticRegression(double gradient_ascent_learning_rate, int gradient_ascent_training_iterations, double gaussian_kernel_sigma) {
		super(gradient_ascent_learning_rate, gradient_ascent_training_iterations);
		this.gaussian_kernel_sigma = gaussian_kernel_sigma;
	}

	@Override
	protected double kernelTransform(FeatureVector x, FeatureVector y) {
		double differenceSquaredNorm = 0.0;

		Iterator<Integer> iter = x.indexIterator();

		while (iter.hasNext()) {
			int index = iter.next();
			double diff = x.get(index) - y.get(index);
			differenceSquaredNorm += diff * diff;
		}

		iter = y.indexIterator();

		while (iter.hasNext()) {
			int index = iter.next();
			if (!x.hasIndex(index)) {
				differenceSquaredNorm += y.get(index) * y.get(index);
			}
		}

		return Math.exp(-1.0 * differenceSquaredNorm / 2.0 / gaussian_kernel_sigma / gaussian_kernel_sigma);
	}
}
