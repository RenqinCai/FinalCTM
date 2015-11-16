import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import structures.MyPriorityQueue;
import structures._RankItem;
import utils.Utils;
import Jama.Matrix;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

public class CTM {
	protected int varMaxIter;
	protected double varConverge;
	protected double emMaxIter;
	protected double emConverge;

	public double[] mu;
	public double[][] inv_cov;
	public double[][] cov;
	public double[][] log_beta;
	public double log_det_cov;
	
	public double[] muStat;
	public double[][] covStat;
	public double[][] word_topic_stats;
	public boolean logSpace;
	
	public double docSize;

	public ArrayList<_Doc> corpus;
	public _Doc[] corpusArray;
	public ArrayList<String> features;
	
	public int vocabulary_size;
	public double[] m_sstat;
	
	public int len1;
	public int len2;
	
	public CTM(int vocabulary_sizeArg, int number_of_topics, int varMaxIterArg, double varConvergeArg, 
			int emMaxIterArg, double emConvergeArg, double cgConvergeArg){
		len1 = number_of_topics;
		len2 = number_of_topics-1;
		
		logSpace = true;
		
//		vocabulary_size = vocabulary_sizeArg;
//		mu = new double[len2];
//		inv_cov = new double[len2][len2];
//		cov = new double[len2][len2];
//		log_beta = new double[len1][vocabulary_size];
//		log_det_cov = 0.0;
		
		
//		docSize = 0.0;
//		muStat = new double[len2];
//		covStat = new double[len2][len2];	
//		word_topic_stats = new double[len1][vocabulary_size];

		features = new ArrayList<String>();
		corpus = new ArrayList<_Doc>();
		varConverge = varConvergeArg;
		varMaxIter = varMaxIterArg;

		emConverge = emConvergeArg;
		emMaxIter = emMaxIterArg;
	}
	

	public CTM(double converge, double beta, ArrayList<_Doc> c, int number_of_topics, int varMaxIterArg, double varConvergeArg, double emMaxIterArg, double emConvergeArg, double cgConvergeArg){
		int len1 = number_of_topics;
		int len2 = number_of_topics-1;
		
		logSpace = true;
		
		corpus = new ArrayList<_Doc>();
		varConverge = varConvergeArg;
		varMaxIter = varMaxIterArg;

		emConverge = emConvergeArg;
		emMaxIter = emMaxIterArg;
	}
	
	public void initModel(){
		System.out.println("initial Model...");
		
		mu = new double[len2];
		inv_cov = new double[len2][len2];
		cov = new double[len2][len2];
		log_beta = new double[len1][vocabulary_size];
		log_det_cov = 0.0;
		
		
		docSize = 0.0;
		muStat = new double[len2];
		covStat = new double[len2][len2];	
		word_topic_stats = new double[len1][vocabulary_size];


		m_sstat = new double[len1];
		
		int initialFlag = 0;
	//	int initialFlag = 1;
		Random r = new Random();
//		long seed = 1115574245;
//		r.setSeed(seed);
//		long t1;
//		t1 = System.currentTimeMillis();
//		r.setSeed(t1);
//		System.out.print("seed "+t1);
		
		
		if(initialFlag ==0){
			
		//	long seed = 1035574983;
		//	r.setSeed(seed);
			//0 is random initial, 1 is corpus initial
			for(int i=0; i<len2; i++){
				mu[i] = 0;
				cov[i][i] = 1.0;
				
			}
			
			double val = 0.0;
			for(int i=0; i<len1; i++){
				double sum = 0.0;
				for(int j=0; j<vocabulary_size; j++){
					val = r.nextDouble()+1.0/100.0;
					//val = r.nextDouble();

					sum += val;
					log_beta[i][j] = val;
					
				}
				
				for(int j=0; j<vocabulary_size; j++){
					log_beta[i][j] = Math.log(log_beta[i][j])-Math.log(sum);
					
				}
			}
				

		}
		else{
			// corpus initial
			long seed = 1115574245;
			r.setSeed(seed);
			
			for(int i=0; i<len2; i++){
				mu[i] = 0;
				cov[i][i] = 1.0;
			}
			
			int corpusSize = corpusArray.length;
			
			for(int i=0; i<len1; i++){
				double sum = 0;
				for(int j=0; j<1; j++){
					double d = corpusSize*r.nextDouble();
					int randomDocID = (int) d;
					System.out.println("docID"+randomDocID);
					
					_Doc doc = corpusArray[randomDocID];
					for(int n=0; n<doc.terms; n++){
						int wid = doc.getIndex(n);
						double value = doc.getValue(n);
						log_beta[i][wid] += value;
					}
				}
				
				for(int n=0; n<vocabulary_size; n++){
					log_beta[i][n] += 1.0 + r.nextDouble();
					sum += log_beta[i][n];
				}
				
				for(int n=0; n<vocabulary_size; n++){
					log_beta[i][n] = Math.log((double)log_beta[i][n]/sum);
				}
			}
			
		}
			Matrix covMatrix = new Matrix(cov);
			Matrix inv_covMatrix = covMatrix.inverse();
			double det_cov = covMatrix.det();
			log_det_cov = Math.log(det_cov);
			inv_cov = inv_covMatrix.getArray();
			// Matrix shrinkageCovMatrix = cov_shrinkage(covMatrix, docSize);
//
//			Matrix inv_covMatrix = covMatrix.inverse();
//			double det_inv_cov = inv_covMatrix.det();
//			log_det_cov = Math.log(det_inv_cov);
//			inv_cov = inv_covMatrix.getArray();
			
		
	
	}
	
	public void initStats(){
		System.out.println("initial stats...");
		
		Arrays.fill(muStat, 0);
		
		for(int i=0; i<len2; i++){
			Arrays.fill(covStat[i], 0);
		}
		
		for(int i=0; i<len1; i++){
			Arrays.fill(word_topic_stats[i], 0);	
		}
		
		docSize = 0;
	}
	
	public void initDoc(_Doc doc, boolean resetParam){
		double phiArg = (double)1/len1;
		//System.out.println("phiArg"+phiArg);
		double zetaArg = 10;
		double nuArg = 10;
		double lambdaArg = 0;
		
		doc.phi = new double[doc.terms][len1];
		for (int n = 0; n <doc.terms; n++) {
			Arrays.fill(doc.phi[n], phiArg);		
		}
		
		
		doc.lambda = new double[len1];
		Arrays.fill(doc.lambda, lambdaArg);
		doc.lambda[len2] = 0;
		
		doc.nu = new double[len1];
		Arrays.fill(doc.nu, nuArg);
		doc.nu[len2] = 0;
		
		doc.zeta = zetaArg;	
		doc.m_topics = new double[len1];
	//	System.out.println("initial doc...");
	}
	
	public double E_step(boolean resetDocParam) {
		String LambdaFile = "Lambda.dat";
		double total = 0;
		try{
		PrintStream out = new PrintStream(new File(LambdaFile));
		
		ArrayList<_Doc> lineSearchFailDoc = new ArrayList<_Doc>();
		
		double curLikelihood = 0.0;
		int docID = 0;
		for(_Doc d: corpus){
			//initDoc(d, resetDocParam);
			curLikelihood = varInference(d, lineSearchFailDoc);
			updateStats(d);
			total += curLikelihood;
//			for(int i=0; i<len1; i++)
//				out.println(d.lambda[i]);
			//System.out.println("variational docID..." + docID);
			docID += 1;
		}
		
		double failTotalLen = 0.0;
		int failDocSize = 0;
		double failAvg = 0.0;
		for(_Doc d: lineSearchFailDoc){
			failTotalLen += d.length;
		}
		
		failDocSize = lineSearchFailDoc.size();
		failAvg = failTotalLen/(double)failDocSize;
		System.out.println("avg doc Len"+failAvg + "fail doc size" + failDocSize);
		
		out.close();
		
		}catch(FileNotFoundException e){
			e.printStackTrace();
		}
		return total;
	}
	
	public double varInference(_Doc doc, ArrayList<_Doc> lineSearchFailDoc){
		int iter = 0;
		double curLikelihood = 0.0;
		double converge = 0.0;
		double oldLikelihood = 0.0;
		
		if(varConverge>0)
			oldLikelihood = calLikelihood(doc);
		
		boolean fail = false;
		do{
			iter += 1;
			opt_zeta(doc);
			//fail is true
			if(opt_lambda(doc)){
				fail = true;
			}
			
			opt_zeta(doc);
			opt_nu(doc);
			
			opt_zeta(doc);
			opt_phi(doc);
			
			if(varConverge>0){
				curLikelihood = calLikelihood(doc);
				
				converge = (oldLikelihood-curLikelihood)/oldLikelihood;
				oldLikelihood = curLikelihood;
				
			}
			//System.out.println("iter"+iter);
		}while((iter<varMaxIter)&&(Math.abs(converge)>varConverge));
		
		if(fail){
			lineSearchFailDoc.add(doc);
		}
		
		return curLikelihood;
	
	}
	
	public double calLikelihood(_Doc doc){
		double likelihood = 0.0;
		
		likelihood += -0.5*log_det_cov;
		likelihood += 0.5*(len2+doc.nu[len2]);
		
		for(int i=0; i<len2; i++){
			likelihood += -(0.5) * doc.nu[i] * inv_cov[i][i];
		}
		
		
		for(int i=0; i<len2; i++){
			for(int j=0; j<len2; j++){
				likelihood += -(0.5) * (doc.lambda[i]-mu[i])*inv_cov[i][j]*(doc.lambda[j] - mu[j]);
			}
		}
		
		for(int i=0; i<len2; i++){
			likelihood += 0.5 * Math.log(doc.nu[i]);		
		}
		
		likelihood += -expect_mult_norm(doc)*doc.length;
		
		for(int n=0; n<doc.terms; n++){
			int wid = doc.getIndex(n);
			double v = doc.getValue(n);
			
			for(int i=0; i<len1; i++){
				likelihood += doc.phi[n][i]*v*(doc.lambda[i]+log_beta[i][wid]-Math
									.log(doc.phi[n][i]));
			}
		}
		
		return likelihood;
		
	}
	
	public double expect_mult_norm(_Doc doc) {
		
		double sum_exp = 0.0;
		double mult_zeta = 0.0;

		for (int i = 0; i <len1; i++) {
			sum_exp += Math.exp(doc.lambda[i] + 0.5 * doc.nu[i]);
		}

		mult_zeta = (1.0 / doc.zeta) * sum_exp - 1 + Math.log(doc.zeta);
		return mult_zeta;

	}
		
	public void updateStats(_Doc doc){	
		for(int i=0; i<len2; i++){
			muStat[i] += doc.lambda[i];
			for(int j=0; j<len2; j++){
				double lilj = doc.lambda[i]*doc.lambda[j];
				
				if(i==j){
					covStat[i][j] += doc.nu[i] + lilj;
				}else{
					covStat[i][j] += lilj;
				}
				
			}
			
		}
		
		for(int n=0; n<doc.terms; n++){
			int wid = doc.getIndex(n);
			double v = doc.getValue(n);
			
			for(int i=0; i<len1; i++){
				word_topic_stats[i][wid] += v*doc.phi[n][i];
				
			}
			
		}
		
		docSize += 1;
		
	}
	
	public void opt_zeta(_Doc doc){
		doc.zeta = 1.0;
		for(int i=0; i<len2; i++){
			doc.zeta += Math.exp(doc.lambda[i] + 0.5 * doc.nu[i]);
		}
	}

	public void opt_phi(_Doc doc){

		double logSum = 0.0;

		for(int n=0; n<doc.terms; n++){
			int wid = doc.getIndex(n);
			double v = doc.getValue(n);

			for(int i=0; i<len1; i++){
				doc.phi[n][i] = log_beta[i][wid]+doc.lambda[i];

			}

			logSum = Utils.logSumOfExponentials(doc.phi[n]);
			for(int i=0; i<len1; i++){
				doc.phi[n][i] = Math.exp(doc.phi[n][i]-logSum);

			}

		}
	}


	public boolean opt_lambda(_Doc doc){
		boolean failSearch = false;
		int[] iflag = {0}, iprint={-1, 3};
		double fValue=0.0;
		int xSize = len2;
		double[] x = new double[len2];
		double[] x_g = new double[len2];
		double[] x_diag = new double[len2];

		double minFValue = 0.0;
		boolean first = true;
		double[] min_x = new double[len2];
		
		for(int i=0; i<len2; i++){
			x[i] = doc.lambda[i];
			x_diag[i] = 0;
			x_g[i] = 0;
			min_x[i] = 0;
		} 

		double eps = 1e-6;

		//Arrays.fill(x_diag, 0);

		try{
			do{
				fValue = calcLambdaFuncGradient(doc, x, x_g);
				
				if (first == true) {
					minFValue = fValue;
					first = false;
					
					for (int i = 0; i < len2; i++) {
						min_x[i] = x[i];
					}
					
				}
				
				if (fValue < minFValue) {
					minFValue = fValue;
					for (int i = 0; i < len2; i++) {
						min_x[i] = x[i];
					}
				}
				LBFGS.lbfgs(xSize, 4, x, fValue, x_g, false, x_diag, iprint,
						eps, 1e-32, iflag);
				
			}while(iflag[0]!=0);

		}catch(ExceptionWithIflag e){
			failSearch = true;
			e.printStackTrace();
		}
		if(iflag[0]==-1)
			System.out.println("docID"+doc.docID);
		for(int i=0; i<len2; i++){
			doc.lambda[i] = min_x[i];
		}
		doc.lambda[len2] = 0;
		return failSearch;
	}

	public double calcLambdaFuncGradient(_Doc doc, double[] x, double[] x_g){
		double[] sum_phi = new double[len2];

		for(int i=0; i<len2; i++){
			for(int n=0; n<doc.terms; n++){
				int wid = doc.getIndex(n);
				double v= doc.getValue(n);

				sum_phi[i] += v*doc.phi[n][i];
			}

		}

		double term1 = 0.0;
		double[] gTerm1 = new double[len2];
		for(int i=0; i<len2; i++){
			term1 += x[i]*sum_phi[i];
			gTerm1[i] = sum_phi[i];

		}

		double term2 = 0.0;
		double[] gTerm2 = new double[len2];
		double[] lambda_mu = new double[len2];
		for(int i=0; i<len2; i++){
			lambda_mu[i] = x[i]-mu[i];
		}

		Matrix inv_covMatrix = new Matrix(inv_cov);
		Matrix lambda_muMatrix = new Matrix(lambda_mu, len2);
		term2 = -0.5* (lambda_muMatrix.transpose().times(
						inv_covMatrix.times(lambda_muMatrix)).get(0, 0));
		for(int i=0; i<len2; i++){
			for(int j=0; j<len2; j++){
				gTerm2[i] -= inv_cov[i][j]*lambda_mu[j];
			}
		}

		double term3=0;
		double[] gTerm3 = new double[len2];
		for(int i=0; i<len2; i++){
			term3 += Math.exp(x[i]+0.5*doc.nu[i]);
			gTerm3[i] = -doc.length*Math.exp(x[i]+doc.nu[i]*0.5)/doc.zeta;
		}
		term3 = -doc.length*(term3/doc.zeta);

		
		double lambdaLikelihood = -(term1+term2+term3);
		for(int i=0; i<len2; i++){
			x_g[i] = -(gTerm1[i]+gTerm2[i]+gTerm3[i]);
		}
		//double lambdaLikelihood = calLikelihood(doc)
		return lambdaLikelihood;
	}

	public void opt_nu(_Doc doc){
		int[] iflag = {0}, iprint={-1, 3};
		double fValue = 0.0;
		int xSize = len2;
		double[] x = new double[len2];
		double[] x_g = new double[len2];
		double[] x_diag = new double[len2];

		for(int i=0; i<len2; i++){
			x[i] = Math.log(doc.nu[i]);
			x_diag[i] = 0;
			x_g[i] = 0;
		}

		try{
			do{
				fValue = calcNuFuncGradient(doc, x, x_g);
				LBFGS.lbfgs(xSize, 4, x, fValue, x_g, false, x_diag, iprint, 1e-6, 1e-32, iflag);
			}while(iflag[0]!=0);

		}catch(ExceptionWithIflag e){
			e.printStackTrace();
		}


		for(int i=0; i<len2; i++){
			doc.nu[i] = Math.exp(x[i]);
		}
	}
	
	public double calcNuFuncGradient(_Doc doc, double[] x, double[] x_g){
		double likelihood = 0.0;

		double term1 = 0.0;
		double term2 = 0.0;
		double term3 = 0.0;

		double[] gTerm1 = new double[len2];
		double[] gTerm2 = new double[len2];
		double[] gTerm3 = new double[len2];

		for(int i=0; i<len2; i++){
			term1 += Math.exp(x[i])*inv_cov[i][i];
			gTerm1[i] = -0.5*Math.exp(x[i])*inv_cov[i][i];
		}
		term1 = -0.5*term1;
		
		for(int i=0; i<len2; i++){
			term2 += Math.exp(doc.lambda[i]+Math.exp(x[i])/2);
			gTerm2[i] = -0.5*Math.exp(x[i])*Math.exp(doc.lambda[i]+Math.exp(x[i])/2)*doc.length/doc.zeta;
		}
		term2 = -doc.length*term2/doc.zeta;

		for(int i=0; i<len2; i++){
			term3 += 0.5*x[i];
			gTerm3[i] = 0.5;
		}

		likelihood = -(term1+term2+term3);
		for(int i=0; i<len2; i++){
			x_g[i] = -(gTerm1[i]+gTerm2[i]+gTerm3[i]);
		}

		return likelihood;

	}

	public void M_step(){
		
		
		//mu
		for(int i=0; i<len2; i++){
			mu[i] = muStat[i]/docSize;
		}
		
		//cov
		for(int i=0; i<len2; i++){
			for(int j=0; j<len2; j++){
				cov[i][j] = covStat[i][j] + docSize * mu[i] * mu[j]- mu[i]* muStat[j] - mu[j] * muStat[i];
				
				cov[i][j] = cov[i][j] / docSize;
			}
			
		}
		Matrix covMatrix = new Matrix(cov);
			
		Matrix inv_covMatrix = covMatrix.inverse();
		double det_cov = covMatrix.det();
		log_det_cov = Math.log(det_cov);
		inv_cov = inv_covMatrix.getArray();
		
		//beta
		for(int i=0; i<len1; i++){
			double sum = Utils.sumOfArray(word_topic_stats[i]);
			for (int n = 0; n < vocabulary_size; n++) {
				log_beta[i][n] = Math.log(word_topic_stats[i][n])-Math.log(sum);
			}
		}
		
	}
	
	

	public void EM(){
		initModel();
		initStats();
		boolean resetDocParam = true;
		int docID = 0;
		for(_Doc d: corpus){
			
			initDoc(d, resetDocParam);
		//	System.out.println("docID..."+ docID);
			docID += 1;
		}
		
		double iter = 0.0;
		double oldTotal = 0.0;
		double curTotal = 0.0;
		double converge = 0.0;
		
		//boolean resetDocParam = true;
		do{
			curTotal = E_step(resetDocParam);
			if(iter >0)
				converge = (oldTotal-curTotal)/oldTotal;
			else
				converge = 1.0;
			if(converge<0){
				resetDocParam = false;
				varMaxIter += 10;	
				System.out.println("E_step not converge");
			}
			else{
				M_step();
				oldTotal = curTotal;
				resetDocParam = true;
				
				//if(iter %10==0){
					System.out.format("Likelihood %.3f at step %s converge to %f...\n", curTotal, iter, converge);
					//	infoWriter.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
				//}
				iter += 1;
				if(converge<emConverge)
					break;
			}
			
			initStats();
			
		}while((iter<emMaxIter)&&(converge<0||converge>emConverge));
	
		finalEst();
	}

	public void readVocabulary(String fileName){
		String line = null;
		
		try{
			FileReader fileReader = new FileReader(fileName);

			BufferedReader bufferReader = new BufferedReader(fileReader);

			while((line=bufferReader.readLine())!=null){
				//corpusSize += 1;
				String[] tokens = line.split("\t");
				//int tokensLen = tokens.length;
				int terms = Integer.parseInt(tokens[1]);
				String word = tokens[0];
				//_Doc d = new _Doc(terms);// to do
				//vocabulary_size += terms;
				
				features.add(word);
				
			}
			//System.out.println("size of features"+features.size()+vocabulary_size);
			vocabulary_size = features.size();
		//	System.out.println("size of features"+features.size()+vocabulary_size);
			bufferReader.close();

		}catch(FileNotFoundException ex){
			System.out.println("unable to open file"+fileName);
		}
		catch(IOException ex){
			ex.printStackTrace();
		}
	}
	
	public void readData(String fileName){
		//ArrayList<_Doc> trainSet = new ArrayList<_Doc>();
		int corpusSize = 0;
		String line = null;
		vocabulary_size = 0;
		try{
			FileReader fileReader = new FileReader(fileName);

			BufferedReader bufferReader = new BufferedReader(fileReader);
			int docID = 0;
			double totalDocLength = 0;
			double maxDocLength = 0;
			double minDocLength = 100.0;
			double avgDocLength = 0;
			while((line=bufferReader.readLine())!=null){
				double docLength = 0.0;
				corpusSize += 1;
				String[] tokens = line.split("\t");

				int tokensLen = tokens.length;
				int terms = Integer.parseInt(tokens[0]);
				//System.out.println("terms"+terms);

				_Doc d = new _Doc(terms, docID);// to do
				//vocabulary_size += terms;
				
				for(int i=1; i<tokensLen; i++){
					String[] words = tokens[i].split(":");
					
					int wordIndex = Integer.parseInt(words[0]);
					double wordValue = Double.parseDouble(words[1]);
				//	System.out.println("words"+words[0]);
				//	System.out.println("words"+words[1]);
					d.setDoc(i, wordIndex, wordValue);
					docLength += wordValue;
				}
				totalDocLength += docLength;
				if(docLength > maxDocLength){
					maxDocLength = docLength;
				}
				if(docLength < minDocLength){
					minDocLength = docLength;
				}
				//trainSet.add(d);
				corpus.add(d);
				docID += 1;
				d.setDocLength(docLength);

			}
			
			
			avgDocLength = totalDocLength/(double)(corpusSize);
			
			corpusArray = new _Doc[corpusSize];
			docID = 0;
			for(_Doc doc: corpus){
				for(int i: doc.wordIndex)
				//System.out.println("wordIndex"+i);;
				corpusArray[docID]=doc;
				docID += 1;			
			}

			System.out.println("maxLen: "+maxDocLength+ " minLen: " + minDocLength + " avgLen: " + avgDocLength);
			bufferReader.close();

		}catch(FileNotFoundException ex){
			System.out.println("unable to open file"+fileName);
		}
		catch(IOException ex){
			ex.printStackTrace();
		}

	}
	
	public void finalEst(){
		
		System.out.print(vocabulary_size);
		String finalLambdaFile = "finalLambda.dat";

		int numLambda = 0;
		try{
			PrintStream out = new PrintStream(new File(finalLambdaFile));
		
			for (_Doc doc : corpus) {
				estThetaInDoc(doc);
				
				if (numLambda >= 7000) {
					continue;
				}
				for(int i=0; i<doc.lambda.length; i++){
					out.print(doc.lambda[i] + "\t");
				}
				out.print("\n");
				numLambda += 1;
			}
			out.flush();
			out.close();
			
			

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	
	protected void estThetaInDoc(_Doc doc){

		double sum = 0;
		Arrays.fill(doc.m_topics, 0);
		
		for (int n = 0; n < doc.terms; n++) {
			int wid = doc.getIndex(n);
			double v = doc.getValue(n);
			for(int i=0; i < len1; i++){
				doc.m_topics[i] += v*doc.phi[n][i];//here should multiply v
			}
		}
		
		sum = Utils.sumOfArray(doc.m_topics);
		for(int i=0; i < len1; i++){
			if (logSpace){
				doc.m_topics[i] = Math.log(doc.m_topics[i]/sum);
				//System.out.println(doc.m_topics[i]);
			}else{
				doc.m_topics[i] = doc.m_topics[i]/sum;
				//System.out.println(doc.m_topics[i]);
			}
		}
		
	}

	public void printTopWords(int k) throws FileNotFoundException {
		String finalBetaFile = "finalBeta.dat";
		PrintStream out = new PrintStream(new File(finalBetaFile));
		
		for(int i=0; i<len1; i++){
			for(int n=0; n<vocabulary_size; n++){
				out.println(log_beta[i][n]);
			}
		}
		Arrays.fill(m_sstat, 0);
		for (_Doc d : corpus) {
			for (int i = 0; i < len1; i++)
				m_sstat[i] += logSpace ? Math.exp(d.m_topics[i])
						: d.m_topics[i];
		}
		Utils.L1Normalization(m_sstat);

		for (int i = 0; i < log_beta.length; i++) {
			MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
					k);
			for (int j = 0; j < vocabulary_size; j++)
				// fVector.add(new _RankItem((j), log_beta[i][j]));
				fVector.add(new _RankItem(features.get(j), log_beta[i][j]));
			System.out.format("Topic%d:%.3f\t", i, m_sstat[i]);
			for (_RankItem it : fVector)
				System.out.format("%s:%.3f\t", it.m_name,
						logSpace ? Math.exp(it.m_value) : it.m_value);
			System.out.println();
		}
		
	}
	
	public static void main(String[] args) throws FileNotFoundException{
		PrintStream out = new PrintStream(new FileOutputStream(
				"output_CTM_newLambda.txt"));
		System.setOut(out);
		
		String formatFileName = "standardFile.dat";
		String vocFileName = "vocab.dat";
		
		int vocabulary_sizeArg = 0; 
		int number_of_topics=30;
		
		int varMaxIterArg=20;
		double varConvergeArg=1e-6;
		
		int emMaxIterArg=1000;
		double emConvergeArg=1e-3;
		
		double cgConvergeArg=1e-6;
		CTM ctmModel = new CTM(vocabulary_sizeArg, number_of_topics,
				varMaxIterArg, varConvergeArg, emMaxIterArg, emConvergeArg,cgConvergeArg);
		ctmModel.readData(formatFileName);
		ctmModel.readVocabulary(vocFileName);
		ctmModel.EM();
		int k = 10;
		ctmModel.printTopWords(k);
	}

}