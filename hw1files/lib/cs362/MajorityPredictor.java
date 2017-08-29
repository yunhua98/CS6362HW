package cs362;

import java.util.HashMap;

public class MajorityPredictor extends Predictor {

	private Label majorityLabel = null;

	public void train(List<Instance> instances) {
		HashMap<String, Integer> labelCounts = new HashMap<String, Integer>();
		int maxNum = 0;
		for (Instance i : instances) {
			String labelName = i.getLabel().toString();
			if (labelCounts.containsKey(labelName)) {
				labelCounts.replace(labelName, labelCounts.get(labelName) + 1);
			} else {
				labelCounts.put(labelName, 1);
			}

			if (labelCounts.get(labelName) > maxNum) {
				maxNum = labelCounts.get(labelName);
				majorityLabel = i.getLabel();
			}
		}
	}
	
	public Label predict(Instance instance) {
		return majorityLabel;
	}
}
