#include "eigenreq.hpp"
#include "population.hpp"

void writeScaledVector( VectorXd x, VectorXd lb, VectorXd ub, int nI, int k );

VectorXd readScaledVector( VectorXd lb, VectorXd ub, int nI, int k );

void readResult( int nI, int k, double *fitness1, double *fitness2 );

void writeAlgVars( int nI, int counteval, Population p, MatrixXd X );

void readAlgVars( int nI, int *counteval, Population &p, MatrixXd &X );
