public double calLikelihood(_Doc doc, double[]x){
		double likelihood = 0.0;
		
		likelihood += -0.5*log_det_cov;
		likelihood += 0.5*(len2+doc.nu[len2]);
		
		for(int i=0; i<len2; i++){
			likelihood += -(0.5) * doc.nu[i] * inv_cov[i][i];
		}
		
		
		double[] gTerm2 = new double[len2];
		double[] lambda_mu = new double[len2];
		for(int i=0; i<len2; i++){
			lambda_mu[i] = x[i]-mu[i];
		}

		Matrix inv_covMatrix = new Matrix(inv_cov);
		Matrix lambda_muMatrix = new Matrix(lambda_mu, len2);

		for(int i=0; i<len2; i++){
			for(int j=0; j<len2; j++){
				gTerm2[i] -= inv_cov[i][j]*lambda_mu[j];
			}
		}

		
		for(int i=0; i<len2; i++){
			for(int j=0; j<len2; j++){
				likelihood += -(0.5) * (x[i]-mu[i])*inv_cov[i][j]*(x[j] - mu[j]);
			}
		}
		
		for(int i=0; i<len2; i++){
			likelihood += 0.5 * Math.log(doc.nu[i]);		
		}
		
		likelihood += -expect_mult_norm(doc)*doc.length;
		
		double[] gTerm1 = new double[len2];
		double[] sum_phi = new double[len2];

		for(int n=0; n<doc.terms; n++){
			int wid = doc.getIndex(n);
			double v = doc.getValue(n);
			
			for(int i=0; i<len1; i++){
				sum_phi[i] += doc.phi[n][i]*v;
				likelihood += doc.phi[n][i]*v*(x[i]+log_beta[i][wid]-Math
									.log(doc.phi[n][i]));
			}
		}

		for(int i=0; i<len2; i++)
			gTerm1 = sum_phi[i];


		double term3=0;
		double[] gTerm3 = new double[len2];
		for(int i=0; i<len2; i++){
			term3 += Math.exp(x[i]+0.5*doc.nu[i]);
			gTerm3[i] = -doc.length*Math.exp(x[i]+doc.nu[i]*0.5)/doc.zeta;
		}
		double sum_exp = 0.0;
		double mult_zeta = 0.0;

		for (int i = 0; i <len1; i++) {
			sum_exp += Math.exp(doc.lambda[i] + 0.5 * doc.nu[i]);
		}

		mult_zeta = (1.0 / doc.zeta) * sum_exp - 1 + Math.log(doc.zeta);
		return mult_zeta;

		return likelihood;
		
	}