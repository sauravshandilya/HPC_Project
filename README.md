# HPC Project



### Instructions:

##### Compilation:

g++ -std=c++11 main.cpp io.cpp calculations.cpp -Wall -Wextra  'pkg-config --libs --cflags opencv'

note: it is ` and not ' in pkg-config ...

##### Execution

./a.out "./input/training.txt" "./input/test.txt" 0.99

./a.out "./input/training.txt" "./input/test.txt" 12
