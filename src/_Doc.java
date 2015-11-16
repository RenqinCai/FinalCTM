public class _Doc{
	public double[] wordCounts;//unique words
	public int[] wordIndex;
	
	public double length;//total words
	public int terms;
	public int docID;
	
	public double[][] phi;
	public double[] lambda;
	public double[] nu;
	public double zeta;
	public double[] m_topics;
	
	public _Doc(int termsArg, int docIDArg){
		terms = termsArg;
		wordCounts = new double[terms];
		wordIndex = new int[terms];
		zeta = 0;
		docID = docIDArg;
		length = 0;
	}
	
	public void setDoc(int arrayIndexArg, int indexArg, double countArg){
		int arrayIndex = arrayIndexArg-1;
		wordCounts[arrayIndex] = countArg;
		wordIndex[arrayIndex] = indexArg;	
		//length += countArg;
	}
	
	public void setDocLength(double lengthArg){
		length = lengthArg;
	}
	
	public int getIndex(int n){
		int index = 0;
		index = wordIndex[n];
		
		return index;
	}
	
	public double getValue(int n){
		double value = 0;
		value = wordCounts[n];
		
		return value;
	}
}