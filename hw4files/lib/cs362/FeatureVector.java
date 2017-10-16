package cs362;

import java.io.Serializable;
import java.util.Iterator;
import java.util.HashMap;

public class FeatureVector implements Serializable {

	private HashMap<Integer, Double> _elements;
	private int _largestIndex;

	public FeatureVector() {
		_elements = new HashMap<Integer, Double>();
	}

	public void add(int index, double value) {
		// TODO Auto-generated method stub
		_elements.put(index, value);
		if (index > _largestIndex) {
			_largestIndex = index;
		}
	}
	
	public double get(int index) {
		// TODO Auto-generated method stub
		return _elements.getOrDefault(index, 0.0);
	}

	public boolean hasIndex(int index) {
		return _elements.containsKey(index);
	}

    public Iterator<Integer> indexIterator() {
        return _elements.keySet().iterator();
    }

    public int size() {
    	return _largestIndex;
    }

}
