#include "individual.hpp"
#include <vector>

class Population 
{
	private:
	int lambdaMO;
	int N;
	vector< Individual > S;

	public:
	Population( int lambdaMOF, int NF );
	Population( const Population &pF );
	void display( bool verbose );
	int getN();
	int getLambdaMO();
	vector< Individual > getS();
	void extremes( double *f1Min, double *f1Max, double *f2Min, double *f2Max );
	void setF( int kF, double fitness1F, double fitness2F );
	void updateStepSize( int k, double pSucc );
	void insert( Individual ind );
	void trim();
};
