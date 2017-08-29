package cs362;

import java.io.Serializable;

public class ClassificationLabel extends Label implements Serializable {

	private int _label;

	public ClassificationLabel(int label) {
		// TODO Auto-generated constructor stub
		this._label = label;
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		return Integer.toString(_label);
	}

}
