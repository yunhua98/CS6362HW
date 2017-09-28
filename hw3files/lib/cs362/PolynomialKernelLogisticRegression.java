package cs362;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

public class PolynomialKernelLogisticRegression extends LinearKernelLogisticRegression {
	private static final long serialVersionUID = 1L;

	private double polynomial_kernel_exponent;

	public PolynomialKernelLogisticRegression(double gradient_ascent_learning_rate, int gradient_ascent_training_iterations, double polynomial_kernel_exponent) {
		super(gradient_ascent_learning_rate, gradient_ascent_training_iterations);
		this.polynomial_kernel_exponent = polynomial_kernel_exponent;
	}

	@Override
	protected double kernelTransform(FeatureVector x, FeatureVector y) {
		return Math.pow((1 + super.kernelTransform(x, y)), polynomial_kernel_exponent);
	}
}
