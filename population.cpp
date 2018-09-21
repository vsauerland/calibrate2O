#include "population.hpp"
#include "auxiliaries.hpp"
#include "testfunctions.hpp"

Population::Population( int lambdaMOF, int NF )
{
	N = NF;
	lambdaMO = lambdaMOF;
}

Population::Population( const Population &pF )
{
	N = pF.N;
	lambdaMO = pF.lambdaMO;
	S = pF.S;
}

void Population::display( bool verbose )
{
	
	for ( int i = 0; i < S.size(); i++ )
	{
		S[ i ].display( verbose );
	}
}

int Population::getLambdaMO()
{
	return( lambdaMO );
}

int Population::getN()
{
	return( N );
}

vector< Individual > Population::getS()
{
	return( S );
}

void Population::extremes( double *f1Min, double *f1Max, double *f2Min, double *f2Max )
{
	VectorXd f1Values = VectorXd::Zero( lambdaMO );
	VectorXd f2Values = VectorXd::Zero( lambdaMO );
	for ( int i = 0; i < lambdaMO; i++ )
	{
		f1Values( i ) = S[ i ].getFitness1();
		f2Values( i ) = S[ i ].getFitness2();
	}
	*f1Min = f1Values.minCoeff();
	*f1Max = f1Values.maxCoeff();
	*f2Min = f2Values.minCoeff();
	*f2Max = f2Values.maxCoeff();
}

void Population::setF( int kF, double fitness1F, double fitness2F )
{
	S[ kF ].setF( fitness1F, fitness2F );
}

void Population::updateStepSize( int k, double pSucc )
{
	double pSuccAv = S[ k ].getPSuccAv();
	double cp = S[ k ].getCp();
	double sigma = S[ k ].getSigma();
	double pSuccTarget = S[ k ].getPSuccTarget();
	double d = S[ k ].getD();
	pSuccAv = ( 1 - cp ) * pSuccAv + cp * pSucc;
	sigma = sigma * exp( ( pSuccAv - pSuccTarget ) / d / ( 1 - pSuccTarget ) );
	S[ k ].setPSuccAv( pSuccAv );
	S[ k ].setSigma( sigma );
}

void Population::insert( Individual ind )
{
	S.push_back( ind );
}

void Population::trim()
{
	int lambda = S.size();
	assert( lambda >= lambdaMO );
	// assemble 2-column matrix M of both fitness values of all individuals
	MatrixXd M = MatrixXd::Zero( lambda, 2 );
	for ( int i = 0; i < lambda; i++ ) M.row( i ) = S[ i ].getF();
	cout << "POPULATION::TRIM matrix of fitness values M:" << endl << M << endl << endl;
	// calculate ranking vector for the individuals (rows of M)
	VectorXd rank;
	rank = hyperVolRank2( M, false ); 
	cout << "POPULATION::TRIM ranks:" << endl << rank << endl << endl;
	// calculate vector of individual indices ordered by ranks:
	VectorXd ix = VectorXd::LinSpaced( lambda, 0, lambda - 1 );
	sortVectors( rank, ix );
	// keep the lambdaMO best ranked individuals in population
	vector<Individual> SNew;
	for ( int i = 0; i < lambdaMO; i++ ) SNew.push_back( S[ ix( i ) ] );
	S.clear();
	for ( int i = 0; i < lambdaMO; i++ ) S.push_back( SNew[ i ] );
}
