My code is in three files

problem2.py
problem5.py
problem6.py

They run with python3, and are only dependent on the libraries random, numpy and mathplotlib.

numpy and mathplotlib can be installed with the following command

pip3 install numpy
pip3 install matplotlib

The code can be run with the following

python3 problem2.py

The commmand above generates the data and plots for problem2

For problem5.py to work the mnist folder must be in the same directory from which the python5.py is being called. The following repl session shows how to generate a plot.

$ python3
Python 3.7.1 (default, Nov  6 2018, 18:45:35)
[Clang 10.0.0 (clang-1000.11.45.5)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import problem5 as pr
>>> pr.genPlot(0, 1)


Finally for problem6.py to work the DanWood.txt file must be in the same directory.
To run the file simply type in the following into the command line

python3 problem6.py

