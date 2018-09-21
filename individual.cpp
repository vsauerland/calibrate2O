#include "individual.hpp"
#include "auxiliaries.hpp"

Individual::Individual
(
	    double cpF, double dF, double pSuccTargetF,
	    double pTreshF, double ccF, double c1F,
	    VectorXd xF, double fitness1F, double pSuccAvF,
	    double sigmaF, VectorXd pcF, MatrixXd CF
)
{
	cp = cpF;
	d = dF;
	pSuccTarget = pSuccTargetF;
	pTresh = pTreshF;
	cc = ccF;
	c1 = c1F;
	x = xF;
	fitness1 = fitness1F;
	fitness2 = 0;
	pSuccAv = pSuccAvF;
	sigma = sigmaF;
	pc = pcF;
	C = CF;
}

Individual::Individual
(
	    double cpF, double dF, double pSuccTargetF,
	    double pTreshF, double ccF, double c1F,
	    VectorXd xF, double fitness1F, double fitness2F,
	    double pSuccAvF, double sigmaF, VectorXd pcF,
	    MatrixXd CF
)
{
	cp = cpF;
	d = dF;
	pSuccTarget = pSuccTargetF;
	pTresh = pTreshF;
	cc = ccF;
	c1 = c1F;
	x = xF;
	fitness1 = fitness1F;
	fitness2 = fitness2F;
	pSuccAv = pSuccAvF;
	sigma = sigmaF;
	pc = pcF;
	C = CF;
}

Individual::Individual( const Individual& ind )
{
	cp = ind.cp;
	d = ind.d;
	pSuccTarget = ind.pSuccTarget;
	pTresh = ind.pTresh;
	cc = ind.cc;
	c1 = ind.c1;
	fitness1 = ind.fitness1;
	fitness2 = ind.fitness2;
	x = ind.x;
	pSuccAv = ind.pSuccAv;
	sigma = ind.sigma;
	pc = ind.pc;
	C = ind.C;
}

void Individual::display( bool verbose )
{
	printf( "\nIndividual: attributes:\n" );
	if ( verbose )
	{
		printf( "1.) OPERATIONAL CONSTANTS:\n" );
		printf( "cp = %f\n", cp );
		printf( "d = %f\n", d );
		printf( "pSuccTarget = %f\n", pSuccTarget );
		printf( "pTresh = %f\n", pTresh );
		printf( "cc = %f\n", cc );
		printf( "c1 = %f\n", c1 );
		printf( "2.) DYNAMIC PARAMETERS:\n" );
	}
	cout << "x:" << endl << x << endl;
	printf( "fitness1 of x = %f\n", fitness1 );
	printf( "fitness2 of x = %f\n", fitness2 );
	if ( verbose )
	{
		printf( "pSuccAv = %f\n", pSuccAv );
		printf( "sigma = %f\n", sigma );
		cout << "pc:" << endl << pc << endl;
		cout << "C:" << endl << C << endl << endl;
	}
}

double Individual::getCp()
{
	return( cp );
}

double Individual::getD()
{
	return( d );
}

double Individual::getPSuccTarget()
{
	return( pSuccTarget );
}

double Individual::getPTresh()
{
	return( pTresh );
}

double Individual::getCc()
{
	return( cc );
}

double Individual::getC1()
{
	return( c1 );
}

VectorXd Individual::getX()
{
	return( x );
}

void Individual::setX( VectorXd xF )
{
	x = xF;
}

double Individual::getFitness1()
{
	return( fitness1 );
}

void Individual::setFitness1( double fitness1F )
{
	fitness1 = fitness1F;
}

double Individual::getFitness2()
{
	return( fitness2 );
}

void Individual::setFitness2( double fitness2F )
{
	fitness2 = fitness2F;
}
	
VectorXd Individual::getF()
{
	VectorXd v = VectorXd::Zero( 2 );
	v( 0 ) = fitness1;
	v( 1 ) = fitness2;
	return( v );
}

void Individual::setF( double fitness1F, double fitness2F )
{
	fitness1 = fitness1F;
	fitness2 = fitness2F;
}

double Individual::getPSuccAv()
{
	return( pSuccAv );
}

void Individual::setPSuccAv( double pSuccAvF )
{
	pSuccAv = pSuccAvF;
}

double Individual::getSigma()
{
	return( sigma );
}

void Individual::setSigma( double sigmaF )
{
	sigma = sigmaF;
}

VectorXd Individual::getPc()
{
	return( pc );
}

void Individual::setPc( VectorXd pcF )
{
	pc = pcF;
}

MatrixXd Individual::getC()
{
	return( C );
}

void Individual::setC( MatrixXd CF )
{
	C = CF;
}

void Individual::updateStepSize( double pSucc )
{
	pSuccAv = ( 1 - cp ) * pSuccAv + cp * pSucc;
	sigma = sigma * exp( ( pSuccAv - pSuccTarget ) / d / ( 1 - pSuccTarget ) );
}

void Individual::updateCovariance( VectorXd xStep )
{
	if ( pSuccAv < pTresh )
	{
		pc = ( 1 - cc ) * pc + sqrt( cc * ( 2 - cc ) ) * xStep;
		C = ( 1 - c1 ) * C + c1 * pc * pc.transpose();
	}
	else
	{
		pc = ( 1 - cc ) * pc;
		C = ( 1 - c1 ) * C + c1 * ( pc * pc.transpose() + cc * ( 2 - cc ) * C );
	}
	C = 0.5 * ( C + C.transpose() ); // enforce symmetry
}

VectorXd Individual::sample()
{
	SelfAdjointEigenSolver<MatrixXd> es( C );
	MatrixXd D = es.eigenvalues().asDiagonal();
	MatrixXd B = es.eigenvectors(); // B = normalized eigenvectors
	cout << "Individual::sample(): C:" << endl << C << endl << endl;
	cout << "Individual::sample(): B:" << endl << B << endl << endl;
	D = D.array().sqrt(); // D contains standard deviations now
	int N = x.size();
	VectorXd y = VectorXd::Zero( N );
	for ( int i = 0; i < N; i++ )
	{
		y( i ) = normaldistribution( 0.0, 1.0 );
//		cout << "Individual::sample y( " << i << " ) = " << y( i ) << endl;
	}
	y = x.array() + sigma * ( B * D * y ).array(); // add mutation, [1] (37)
	return( y );
}

MatrixXd Individual::sampleSet( int lambda )
{
	SelfAdjointEigenSolver<MatrixXd> es( C );
	MatrixXd D = es.eigenvalues().asDiagonal();
	MatrixXd B = es.eigenvectors(); // B = normalized eigenvectors
	D = D.array().sqrt(); // D contains standard deviations now
	int N = x.size();
	MatrixXd arz = MatrixXd::Zero( N, lambda );
	MatrixXd y = MatrixXd::Zero( N, lambda );
	for ( int k = 0; k < lambda; k++ )
	{
		for ( int i = 0; i < N; i++ )
		{
			arz( i, k ) = normaldistribution( 0.0, 1.0 );
		}
		y.col( k ) = x.array() + sigma * ( B * D * arz.col( k ) ).array(); // add mutation, [1] (37)
	}
//	cout << "C" << endl << C << endl << endl;
//	cout << "B" << endl << B << endl << endl;
//	cout << "D" << endl << D << endl << endl;
//	cout << "BDDB'" << endl << B * D * D * B.transpose() << endl << endl;
//	cout << "arz" << endl << arz << endl << endl;
	return( y );
}
