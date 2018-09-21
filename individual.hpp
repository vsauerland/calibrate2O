#include "eigenreq.hpp"

class Individual
{
	private:
	// constants used for update operations
	double cp;
	double d;
	double pSuccTarget;
	double pTresh;
	double cc;
	double c1;
	// dynamic parmeters
	VectorXd x; // current distribution mean
	double fitness1; // (first) objective value of x
	double fitness2; // (optional) second objective value of x
	double pSuccAv; // average sampling success rate
	double sigma; // distribution variance factor
	VectorXd pc; // evolution path
	MatrixXd C; // covariance matrix
	
	public:
	// construct without setting 2nd objective function value
	Individual( double cpF, double dF, double pSuccTargetF,
		    double pTreshF, double ccF, double c1F,
		    VectorXd xF, double fitness1, double pSuccAvF,
		    double sigmaF, VectorXd pcF, MatrixXd CF );
	// construct with explicit 2nd objective function value
	Individual( double cpF, double dF, double pSuccTargetF,
		    double pTreshF, double ccF, double c1F,
		    VectorXd xF, double fitness1, double fitness2,
		    double pSuccAvF, double sigmaF, VectorXd pcF,
		    MatrixXd CF );
	Individual( const Individual& ind );
	void display( bool verbose );
	double getCp();
	double getD();
	double getPSuccTarget();
	double getPTresh();
	double getCc();
	double getC1();
	VectorXd getX();
	void setX( VectorXd xF );
	double getFitness1();
	void setFitness1( double fitness1F );
	double getFitness2();
	void setFitness2( double fitness2F );
	VectorXd getF();
	void setF( double fitness1F, double fitness2F );
	double getPSuccAv();
	void setPSuccAv( double pSuccAvF );
	double getSigma();
	void setSigma( double sigmaF );
	VectorXd getPc();
	void setPc( VectorXd pcF );
	MatrixXd getC();
	void setC( MatrixXd CF );
	void updateStepSize( double pSucc );
	void updateCovariance( VectorXd xStep );
	VectorXd sample();
	MatrixXd sampleSet( int lambda );
};
