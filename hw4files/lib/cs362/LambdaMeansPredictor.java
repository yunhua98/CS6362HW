package cs362;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.HashMap;
import java.util.HashSet;

public class LambdaMeansPredictor extends Predictor {
	private static final long serialVersionUID = 1L;

	private double cluster_lambda;
	private int clustering_training_iterations;
	private ArrayList<FeatureVector> clusters;
	// private int[] cluster_assignments;
	private HashMap<Integer, HashSet<Integer>> cluster_to_instances;

	public LambdaMeansPredictor(double cluster_lambda, int clustering_training_iterations) {
		this.cluster_lambda = cluster_lambda;
		this.clustering_training_iterations = clustering_training_iterations;
		this.clusters = new ArrayList<FeatureVector>();
	}

	public void train(List<Instance> instances) {
		// cluster_assignments = new int[instances.size()];
		cluster_to_instances = new HashMap<Integer, HashSet<Integer>>();
		FeatureVector prototype = new FeatureVector();

		for (Instance i : instances) {
			FeatureVector f = i.getFeatureVector();
			Iterator<Integer> index_iter = f.indexIterator();
			while (index_iter.hasNext()) {
				int index = index_iter.next();
				prototype.add(index, f.get(index) + prototype.get(index));
			}
		}

		int m = instances.size();
		Iterator<Integer> index_iter = prototype.indexIterator();
		while (index_iter.hasNext()) {
			int index = index_iter.next();
			prototype.add(index, prototype.get(index) / m);
		}

		clusters.add(prototype);

		if (cluster_lambda == 0.0) {
			double diff;
			for (Instance i : instances) {
				FeatureVector f = i.getFeatureVector();
				index_iter = prototype.indexIterator();
				while (index_iter.hasNext()) {
					int index = index_iter.next();
					diff = prototype.get(index) - f.get(index);
					cluster_lambda += diff * diff;
				}
			}
			cluster_lambda /= m;
		}

		System.out.println("Lambda: " + cluster_lambda);

		for (int iterations = 0; iterations < clustering_training_iterations; ++iterations) {
			System.out.println("Iteration: " + iterations);
			cluster_to_instances.clear();

			// E Step
			for (int i = 0; i < instances.size(); ++i) {
				double minDist = Double.MAX_VALUE;
				int minCluster = -1;

				FeatureVector x = instances.get(i).getFeatureVector();
				Iterator<Integer> x_iter;
				double dist, diff;
				int index;
				for (int j = 0; j < clusters.size(); ++j) {
					FeatureVector mu = clusters.get(j);
					x_iter = x.indexIterator();
					dist = 0.0;
					while (x_iter.hasNext()) {
						index = x_iter.next();
						diff = x.get(index) - mu.get(index);
						dist += diff * diff;
					}

					Iterator<Integer> mu_iter = mu.indexIterator();
					while (mu_iter.hasNext()) {
						index = mu_iter.next();
						if (!x.hasIndex(index)) {
							dist += mu.get(index) * mu.get(index);
						}
					}

					if (dist < minDist) {  // Strictly less than so lowest index cluster wins ties
						minDist = dist;
						minCluster = j;
					}
				}

				int assigned_cluster = minCluster;
				if (minDist > cluster_lambda) {
					assigned_cluster = clusters.size();
					clusters.add(x);
				}

				if (!cluster_to_instances.containsKey(assigned_cluster)) {
					cluster_to_instances.put(assigned_cluster, new HashSet<Integer>());
				}

				cluster_to_instances.get(assigned_cluster).add(i);
			}

			// M Step
			for (int i = 0; i < clusters.size(); ++i) {
				if (!cluster_to_instances.containsKey(i)) {
					clusters.set(i, new FeatureVector());
					continue;
				}

				FeatureVector mu = new FeatureVector();
				for (Integer inst_num : cluster_to_instances.get(i)) {
					FeatureVector x = instances.get(inst_num).getFeatureVector();
					Iterator<Integer> x_iter = x.indexIterator();
					while (x_iter.hasNext()) {
						int index = x_iter.next();
						mu.add(index, mu.get(index) + x.get(index));
					}
				}

				Iterator<Integer> mu_iter = mu.indexIterator();
				int num = cluster_to_instances.get(i).size();
				while (mu_iter.hasNext()) {
					int index = mu_iter.next();
					mu.add(index, mu.get(index) / num);
				}

				clusters.set(i, mu);
			}
		}

		// for (int cluster_num = 0; cluster_num < clusters.size(); ++cluster_num) {
		// 	 for (Integer inst_num : cluster_to_instances.get(i)) {
		// 	 	 cluster_assignments[inst_num] = cluster_num;
		// 	 }
		//  }

	}
	
	public Label predict(Instance instance) {
		int minCluster = -1;
		double minDist = Double.MAX_VALUE;
		FeatureVector x = instance.getFeatureVector();
		for (int i = 0; i < clusters.size(); ++i) {
			FeatureVector mu = clusters.get(i);
			Iterator<Integer> x_iter = x.indexIterator();
			double dist = 0.0;
			while (x_iter.hasNext()) {
				int index = x_iter.next();
				double diff = x.get(index) - mu.get(index);
				dist += diff * diff;
			}

			Iterator<Integer> mu_iter = mu.indexIterator();
			while (mu_iter.hasNext()) {
				int index = mu_iter.next();
				if (!x.hasIndex(index)) {
					dist += mu.get(index) * mu.get(index);
				}
			}

			if (dist < minDist) {  // Strictly less than so lowest index cluster wins ties
				minDist = dist;
				minCluster = i;
			}
		}

		return new ClassificationLabel(minCluster);
	}

}
