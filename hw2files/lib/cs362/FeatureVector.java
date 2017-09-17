package cs362;

import java.io.Serializable;
import java.util.Iterator;
import java.util.TreeMap;

public class FeatureVector implements Serializable {

	private TreeMap<Integer, Double> _elements;

	public FeatureVector() {
		_elements = new TreeMap<Integer, Double>();
	}

	public void add(int index, double value) {
		// TODO Auto-generated method stub
		_elements.put(index, value);
	}
	
	public double get(int index) {
		// TODO Auto-generated method stub
		return _elements.get(index);
	}

    public Iterator<Integer> indexIterator() {
        return _elements.keySet().iterator();
    }

    public int size() {
    	return _elements.lastKey();
    }

}
