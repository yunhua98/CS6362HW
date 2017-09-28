package cs362;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

public class KernelLogisticRegression extends Predictor {
	private static final long serialVersionUID = 1L;

	private double gradient_ascent_learning_rate;
	private int gradient_ascent_training_iterations;
	private double[] alphas;
	private double[][] gramMatrix;
	private List<Instance> instances;

	public KernelLogisticRegression(double gradient_ascent_learning_rate, int gradient_ascent_training_iterations) {
		this.gradient_ascent_learning_rate = gradient_ascent_learning_rate;
		this.gradient_ascent_training_iterations = gradient_ascent_training_iterations;
	}

	public void train(List<Instance> instances) {
		this.instances = instances;
		alphas = new double[instances.size()];
		double[] gradients = new double[instances.size()];
		gramMatrix = new double[instances.size()][instances.size()];
		
		precompute();

		for (int iterations = 0; iterations < gradient_ascent_training_iterations; ++iterations) {
			System.out.println("Iteration: " + iterations);
			for (int k = 0; k < instances.size(); ++k) {
				FeatureVector featuresK = instances.get(k).getFeatureVector();
				double gradient = 0.0;

				for (int i = 0; i < instances.size(); ++i) {
					FeatureVector featuresI = instances.get(i).getFeatureVector();
					double labelI = Double.parseDouble(instances.get(i).getLabel().toString());
					
					double z = 0.0;
					for (int j = 0; j < instances.size(); ++j) {
						FeatureVector featuresJ = instances.get(j).getFeatureVector();
						z -= alphas[j] * gramMatrix[j][i];  // negative z
					}

					if (labelI == 1.0) {  // if (labelI == 1.0)
						gradient += gramMatrix[i][k] / (1.0 + Math.exp(-1 * z));
					} else {  // if (labelI == 0.0)
						gradient += -1.0 * gramMatrix[i][k] / (1.0 + Math.exp(z));
					}
				}

				gradients[k] = gradient;
			}

			for (int j = 0; j < alphas.length; ++j) {
				alphas[j] += gradient_ascent_learning_rate * gradients[j];
			}
 		}
	}
	
	public Label predict(Instance instance) {
		double z = 0.0;
		FeatureVector features = instance.getFeatureVector();

		for (int j = 0; j < alphas.length; ++j) {
			z -= alphas[j] * kernelTransform(instances.get(j).getFeatureVector(), features);  // negative z
		}

		return new ClassificationLabel((1 / (1 + Math.exp(z)) >= 0.5 ? 1 : 0));
	}

	private void precompute() {
		for (int i = 0; i < instances.size(); ++i) {
			for (int j = 0; j <= i; ++j) {
				gramMatrix[i][j] = kernelTransform(instances.get(i).getFeatureVector(), instances.get(j).getFeatureVector());
				gramMatrix[j][i] = gramMatrix[i][j];
			}
		}
	}

	protected double kernelTransform(FeatureVector x, FeatureVector y) {
		System.out.println("This line should never print");
		return 0.0;  // Should be overridden
	}

}
