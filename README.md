# python-performance-tutorial
A tutorial for optimizing CPU bound applications in python

## Introduction  

In this tutorial we will present a simple way to optimize CPU bound python programs. CPU bound applications are defined as applications whose runtime performance is mainly dominated by the number of CPU cycles for the program operations in contrast to memory or I/O bound applications for which the CPU is mainly idle. Such applications include numerical computing algorithms, feature extraction modules etc.

The goal of this tutorial is not to achieve the absolutely best performance, but to showcase that Python modules can become viable enough for such tasks and avoid rewrites in lower level languages (e.g. C/C++).

What we will cover:

1. Python profiling using [flame graphs](http://www.brendangregg.com/flamegraphs.html)  
2. Vectorization of operations to take full advantage of NumPy's performance  
3. Implementing slow functions in Cython  

As a working example we will study parts and optimize parts of a suboptimal implementation of the Multidimensional scaling algorithm. To get a sense of the effectiveness of this techniques, initial benchmarks show ~30x-40x speedup in execution time.

## CPU Flame Graphs

The first step when optimizating an application is profiling. By profiling we can focus on parts of the code that are more time consuming and focus on them to get maximum performance gains by spending minimum effort. There are many profiling tools for Python (cProfile, PyCharm's integrated profiler etc.) but I recommend using [Uber's pyflame](https://github.com/uber/pyflame). Pyflame is fast, simple to use and can generate flamegraph parseable profiling data. 

Flame Graphs are a visualization for sampled stack traces, which allows hot code-paths to be identified quickly. Here's an example:
![Alt text](https://raw.githubusercontent.com/georgepar/python-performance-tutorial/master/example.svg?sanitize=true)
<img src="https://raw.githubusercontent.com/georgepar/python-performance-tutorial/master/example.svg?sanitize=true">
