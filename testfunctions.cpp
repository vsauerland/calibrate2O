#include <iostream>
#include "testfunctions.hpp"

TTestFunctions::TTestFunctions( int n, int seed )
{
	double pi = 3.141592653589793;
	srand( seed );
	MatrixXd M = MatrixXd::Zero( n, n );
	// generate a standard normal distributed matrix:
	for ( int i = 0; i < n; i++ ) for ( int j = 0; j < n; j++ )
	{
		double r1 = ( double )rand() / RAND_MAX;
		double r2 = ( double )rand() / RAND_MAX;
		M( i, j ) = sqrt( -2 * log( r1 ) ) * sin( 2 * pi * r2 );
	}
	// derive first orhtonormal matrix:
	O1 = M.householderQr().householderQ();
	// generate another standard normal distributed matrix
	for ( int i = 0; i < n; i++ ) for ( int j = 0; j < n; j++ )
	{
		double r1 = ( double )rand() / RAND_MAX;
		double r2 = ( double )rand() / RAND_MAX;
		M( i, j ) = sqrt( -2 * log( r1 ) ) * sin( 2 * pi * r2 );
	}
	// derive second orthonormal matrix:
	O2 = M.householderQr().householderQ();
}

void TTestFunctions::display()
{
	cout << "TTestFunctions:" << endl;
	cout << "first rotation matrix:" << endl << O1 << endl;
	cout << "second rotation matrix:" << endl << O2 << endl << endl;
}

double TTestFunctions::griewank( VectorXd x )
{
	int n = x.size();
	double obj = 0;
	double amp_scale = 0.25;
	double freq_scale = 63.24555320336759; // sqrt( 4000 )
	for ( int j = 0; j < n; j++ )
	{
		obj = obj + x( j ) * x( j );
	}
	double prod = 1;
	for ( int j = 0; j < n; j++ )
	{
		prod = prod * cos( freq_scale / ( j + 1 ) * x( j ) );
	}
	obj = obj + amp_scale * ( 1 - prod );
	return( obj );
}

double TTestFunctions::rosenbrock( VectorXd x )
{
	int n = x.size();
	double obj = 0;
	double q1, q2, q3, d;
	for ( int j = 0; j < n - 1; j++ )
	{
		q1 = x( j ) * x( j );
		q2 = q1 + 1 - 2 * x( j );
		d = x( j + 1 ) - q1;
		q3 = d * d;
		obj = obj + q2 + 100 * q3;
	}
	return( obj );
}

double TTestFunctions::sphere( VectorXd x )
{
	return( ( x.array() * x.array() ).sum() );
}

double TTestFunctions::ssphere( VectorXd x )
{
	return( pow( ( x.array() * x.array() ).sum(), ( double )0.5 ) );
}

double TTestFunctions::schwefel( VectorXd x )
{
	int n = x.size();
	double r = 0;
	for ( int i = 1; i <= n; i++ )
	{
		r = r + ( x.head( i ).sum() ) * ( x.head( i ).sum() );
	}
	return( r );
}

double TTestFunctions::cigar( VectorXd x )
{
	int n = x.size();
	return( x( 0 ) * x( 0 ) + 1e6 * ( x.tail( n - 1 ).array() * x.tail( n - 1 ).array() ).sum() );
}  

double TTestFunctions::cigtab( VectorXd x )
{
	int n = x.size();
	return( x( 0 ) * x( 0 ) + 1e8 * x( n - 1 ) * x( n - 1 ) + 1e4 * ( x.segment( 1, n - 2 ).array() * x.segment( 1, n - 2 ).array() ).sum() );
}  

// rotatet variations of cigtab for 2-objective optimization according to [1]
// for problem CIGTAB1 in [1] we have f1=cigtabRotateA, f2=cigtabRotateB
// for problem CIGTAB2 in [1] we have f1=cigtabRotateA, f2=cigtabRotateC
double TTestFunctions::cigtabRotateA( VectorXd x )
{
	int n = x.size();
	double a = 1000;
	double b = 1000;
	VectorXd y = O1 * x;
	return( ( y( 0 ) * y( 0 ) + b * b * y( n - 1 ) * y( n - 1 ) + b * ( y.segment( 1, n - 2 ).array() * y.segment( 1, n - 2 ).array() ).sum() ) / ( a * a * n )  );
}  
double TTestFunctions::cigtabRotateB( VectorXd x )
{
	int n = x.size();
	double a = 1000;
	double b = 1000;
	VectorXd y = ( O1 * x  ) - VectorXd::LinSpaced( n, 2, 2 );
	return( ( y( 0 ) * y( 0 ) + b * b * y( n - 1 ) * y( n - 1 ) + b * ( y.segment( 1, n - 2 ).array() * y.segment( 1, n - 2 ).array() ).sum() ) / ( a * a * n )  );
}  
double TTestFunctions::cigtabRotateC( VectorXd x )
{
	int n = x.size();
	double a = 1000;
	double b = 1000;
	VectorXd y = ( O2 * x ) - VectorXd::LinSpaced( n, 2, 2 );
	return( ( y( 0 ) * y( 0 ) + b * b * y( n - 1 ) * y( n - 1 ) + b * ( y.segment( 1, n - 2 ).array() * y.segment( 1, n - 2 ).array() ).sum() ) / ( a * a * n )  );
}  

double TTestFunctions::tablet( VectorXd x )
{
	int n = x.size();
	return ( 1e6 * x( 0 ) * x( 0 ) + ( x.tail( n - 1 ).array() * x.tail( n - 1 ).array() ).sum() );
}

double TTestFunctions::elli( VectorXd x )
{
	int n = x.size();
	VectorXd v = VectorXd::LinSpaced( n, 0, 1 );
	for ( int i = 0; i < n; i++ ) v( i ) = pow( ( double )1e6, v( i ) );
	VectorXd r = v.transpose() * ( x.array() * x.array() ).matrix();
	return( r( 0 ) / ( 1e6 * n ) );
}

// rotatet variations of elli for 2-objective optimization according to [1]
// for problem ELLI1 in [1] we have f1=elliRotateA, f2=elliRotateB
// for problem ELLI2 in [1] we have f1=elliRotateA, f2=elliRotateC
double TTestFunctions::elliRotateA( VectorXd x )
{
	int n = x.size();
	double a = 1000;
	double b = 1000;
	VectorXd y = O1 * x;
	VectorXd v = VectorXd::LinSpaced( n, 0, 1 );
	for ( int i = 0; i < n; i++ ) v( i ) = pow( b, 2 * v( i ) );
	VectorXd r = v.transpose() * ( y.array() * y.array() ).matrix();
	return ( r( 0 ) / ( a * a * n ) );
}
double TTestFunctions::elliRotateB( VectorXd x )
{
	int n = x.size();
	double a = 1000;
	double b = 1000;
	VectorXd y = ( O1 * x  ) - VectorXd::LinSpaced( n, 2, 2 );
	VectorXd v = VectorXd::LinSpaced( n, 0, 1 );
	for ( int i = 0; i < n; i++ ) v( i ) = pow( b, 2 * v( i ) );
	VectorXd r = v.transpose() * ( y.array() * y.array() ).matrix();
	return ( r( 0 ) / ( a * a * n ) );
}
double TTestFunctions::elliRotateC( VectorXd x )
{
	int n = x.size();
	double a = 1000;
	double b = 1000;
	VectorXd y = ( O2 * x ) - VectorXd::LinSpaced( n, 2, 2 );
	VectorXd v = VectorXd::LinSpaced( n, 0, 1 );
	for ( int i = 0; i < n; i++ ) v( i ) = pow( b, 2 * v( i ) );
	VectorXd r = v.transpose() * ( y.array() * y.array() ).matrix();
	return ( r( 0 ) / ( a * a * n ) );
}

double TTestFunctions::plane( VectorXd x )
{
	return( x( 0 ) );
}  

double TTestFunctions::twoaxes( VectorXd x )
{
	int n = x.size();
	int k = floor( n / 2 );
	VectorXd v1 = x.head( k ).array() * x.head( k ).array();
	VectorXd v2 = x.tail( n - k ).array() * x.tail( n - k ).array();
	return( v1.sum() + 1e6 * v2.sum() );
}

double TTestFunctions::parabR( VectorXd x )
{
	int n = x.size();
	VectorXd v = x.tail( n - 1 ).array() * x.tail( n - 1 ).array();
	return( -x( 0 ) + 100 * v.sum() );
}

double TTestFunctions::sharpR( VectorXd x )
{
	int n = x.size();
	return( -x( 0 ) + 100 * x.tail( n - 1 ).norm() );
}

double TTestFunctions::diffpow( VectorXd x )
{
	int n = x.size();
	VectorXd v = VectorXd::LinSpaced( n, 0, 10 );
	v = 2 + v.array();
	for ( int i = 0; i < n; i++ ) v( i ) = pow( abs( x( i ) ), v( i ) );
	return( v.sum() );
} 

double TTestFunctions::rastrigin10( VectorXd x )
{
	int n = x.size();
	double pi = 3.141592653589793;
	VectorXd v = VectorXd::LinSpaced( n, 0, 1 );
	for ( int i = 0; i < n; i++ ) v( i ) = pow( ( double )10, v( i ) );
	v = v.array() * x.array();
	VectorXd w( n );
	for ( int i = 0; i < n; i++ ) w( i ) = 10 * cos( 2 * pi * v( i ) );
	return( 10 * n + ( ( v.array() * v.array() ).matrix() - w ).sum() );
}

double TTestFunctions::scaledSine( VectorXd x )
{
	assert( x.size() == 2 );
//	return( 1 + x( 0 ) * sin( x( 1 ) ) );
	// introduced a fancy radial distortion
	double r = x( 0 ) * x( 0 ) + ( 1 - x( 0 ) ) * sin( 20 * x ( 0 ) );
	return( 1 + r * sin( x( 1 ) ) );
}

double TTestFunctions::scaledCosine( VectorXd x )
{
	assert( x.size() <= 3 );
//	return( 1 + x( 0 ) * cos( x( 1 ) ) );
	// introduced a fancy radial distortion
	double r = x( 0 ) * x( 0 ) + ( 1 - x( 0 ) ) * sin( 20 * x ( 0 ) );
	return( 1 + r * cos( x( 1 ) ) );
}

double TTestFunctions::f( const string &fName, VectorXd x )
{
	assert( x.size() == O1.rows() );
	if ( fName == "rastrigin10" ) return( rastrigin10( x ) );
	else if ( fName == "griewank" ) return( griewank( x ) );
	else if ( fName == "rosenbrock" ) return( rosenbrock( x ) );
	else if ( fName == "sphere" ) return( sphere( x ) );
	else if ( fName == "ssphere" ) return( ssphere( x ) );
	else if ( fName == "schwefel" ) return( schwefel( x ) );
	else if ( fName == "cigar" ) return( cigar( x ) );
	else if ( fName == "cigtab" ) return( cigtab( x ) );
	else if ( fName == "cigtabRotateA" ) return( cigtabRotateA( x ) );
	else if ( fName == "cigtabRotateB" ) return( cigtabRotateB( x ) );
	else if ( fName == "cigtabRotateC" ) return( cigtabRotateC( x ) );
	else if ( fName == "tablet" ) return( tablet( x ) );
	else if ( fName == "elli" ) return( elli( x ) );
	else if ( fName == "elliRotateA" ) return( elliRotateA( x ) );
	else if ( fName == "elliRotateB" ) return( elliRotateB( x ) );
	else if ( fName == "elliRotateC" ) return( elliRotateC( x ) );
	else if ( fName == "plane" ) return( plane( x ) );
	else if ( fName == "twoaxes" ) return( twoaxes( x ) );
	else if ( fName == "parabR" ) return( parabR( x ) );
	else if ( fName == "sharpR" ) return( sharpR( x ) );
	else if ( fName == "diffpow" ) return( diffpow( x ) );
	else if ( fName == "scaledSine" ) return( scaledSine( x ) );
	else if ( fName == "scaledCosine" ) return( scaledCosine( x ) );
	else return( rosenbrock( x ) );
}


/* -------------------------------- Literature --------------------------------

[1] C. Igel, N. Hansen, and S. Roth (2007). Covariance Matrix Adaption for
    Multi-objective Optimization. Evolutionary Computation 15(1). 28 pages.

---------------------------------------------------------------------------- */
