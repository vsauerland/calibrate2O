CCC = g++
CC11 = gnu++0x
EIGEN = -I /home/sunip173/eigen-eigen-36fd1ba04c12/Eigen/

CCFLAGS = -O -std=$(CC11) 

PROGRAMS = cmaes_2o testCase

all:
	make $(PROGRAMS)

clean :
	/bin/rm -rf $(PROGRAMS)
	/bin/rm -rf *.o *~ *.txt *.log *.out

cmaes_2o: testfunctions.o auxiliaries.o individual.o population.o io.o cmaes_2o.o
	$(CCC) $(CCFLAGS) $(EIGEN) testfunctions.o auxiliaries.o individual.o population.o io.o cmaes_2o.o -o cmaes_2o
testCase: testfunctions.o testCase.o
	$(CCC) $(CCFLAGS) $(EIGEN) testfunctions.o testCase.o -o testCase

testfunctions.o: testfunctions.cpp 
	$(CCC) -c $(CCFLAGS) $(EIGEN) testfunctions.cpp -o testfunctions.o
auxiliaries.o: auxiliaries.cpp
	$(CCC) -c $(CCFLAGS) $(EIGEN) auxiliaries.cpp -o auxiliaries.o
individual.o: individual.cpp
	$(CCC) -c $(CCFLAGS) $(EIGEN) individual.cpp -o individual.o
population.o: population.cpp
	$(CCC) -c $(CCFLAGS) $(EIGEN) population.cpp -o population.o
io.o: io.cpp
	$(CCC) -c $(CCFLAGS) $(EIGEN) io.cpp -o io.o
cmaes_2o.o: cmaes_2o.cpp
	$(CCC) -c $(CCFLAGS) $(EIGEN) cmaes_2o.cpp -o cmaes_2o.o
testCase.o: testCase.cpp
	$(CCC) -c $(CCFLAGS) $(EIGEN) testCase.cpp -o testCase.o
