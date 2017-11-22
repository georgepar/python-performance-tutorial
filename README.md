# python-performance-tutorial
A tutorial for optimizing CPU bound applications in python

## Introduction  

In this tutorial we will present a simple way to optimize CPU bound python programs. CPU bound applications are defined as applications whose runtime performance is mainly dominated by the number of CPU cycles for the program operations in contrast to memory or I/O bound applications for which the CPU is mainly idle. Such applications include numerical computing algorithms, feature extraction modules etc.

The goal of this tutorial is not to achieve the absolutely best performance, but to showcase that Python modules can become viable enough for such tasks and avoid rewrites in lower level languages (e.g. C/C++).

What we will cover:

1. Python profiling using [flame graphs](http://www.brendangregg.com/flamegraphs.html)  
2. Vectorization of operations to take full advantage of NumPy's performance  
3. Implementing slow functions in Cython  

As a working example we will study parts and optimize parts of a suboptimal implementation of the Multidimensional scaling algorithm. The initial implementation takes about 90 seconds to run in a predefined configuration that we will use as a benchmark.

To get a sense of the effectiveness of this techniques, initial benchmarks show ~30x-40x speedup in execution time.

## CPU Flame Graphs

The first step when optimizating an application is profiling. By profiling we can focus on parts of the code that are more time consuming and focus on them to get maximum performance gains by spending minimum effort. There are many profiling tools for Python (cProfile, PyCharm's integrated profiler etc.) but I recommend using [Uber's pyflame](https://github.com/uber/pyflame). Pyflame is fast, simple to use and can generate flamegraph parseable profiling data. 

Flame Graphs are a visualization for sampled stack traces, which allows hot code-paths to be identified quickly. Here's an example. Github markdown doesn't allow full rendering of SVGs for security purposes, but if you clone the repo and open the SVGs in your browser the boxes should be clickable and you should be able to zoom in/out:

![Alt text](./example.svg)

Let's inspect the visualization and try to make sense of it.

1. Each box represents a function in the stack  
2. If a box (f1) is on top of another (f2) it means that f2 calls f1  
3. It follows that on the vertical axis we can see the program's call stack  
4. The width of a box shows the time it was on-CPU  

TL;DR: Find the wider boxes that correspond to our code and optimize them first.  

You can create a flamegraph for an python program by
1. installing pyflame and FlameGraph:  
```bash
# install pyflame
apt-get install autoconf automake autotools-dev g++ pkg-config python-dev python3-dev libtool make
git clone https://github.com/uber/pyflame
cd pyflame
./autogen.sh
./configure
make
make install

# install flamegraph
git clone https://github.com/brendangregg/FlameGraph ~/FlameGraph
```
2. running the profiler:  
```bash
pyflame -s 10 -r 0.01 -o perf.data -t python my_script.py
```
3. generating and inspecting the svg:
```bash
~/FlameGraph/flamegraph.pl perf.data > perf.svg
firefox perf.svg
```

Here's the flamegraph for our starting implementation of the MDS algorithm: 
![Alt text](./starting_implementation.svg)

## Vectorizing operations with NumPy  

We can see from the above image that over 90% of the program runtime is spent in function called `compute_distance_matrix`:

Let's take a look at the code:
```python
def compute_distance_matrix(Xs,D_current=None,i=None):
    if i is None:
        D_current = np.zeros((Xs.shape[0],Xs.shape[0]))
            for k in range(len(Xs)):
                for l in range(len(Xs)):
                    D_current[k,l] = np.linalg.norm(Xs[k]-Xs[l],2)
    else:
        for k in range(len(Xs)):
            D_current[k,i]=D_current[i,k] = np.linalg.norm(Xs[k]-Xs[i],2)
    return D_current
```

This function essentially computes the distance of each point (row) in `Xs` from a specified row `Xs[i]` and modifies the appropriate column/row in the `D_current` distance matrix. Notice that `D_current` is symmetric.

It is well documented that NumPy operations don't have optimal performance when used in for loops. Instead all operations should be vectorized as follows:

```python
def compute_distance_matrix(xs, d_current, i):
    idx = np.arange(xs.shape[0])
    norm = NORM2(xs - xs[i], ord=2, axis=1)
    d_current[idx, i] = norm
    d_current[i, idx] = norm
    return d_current
```

In the same spirit, we can rewrite the `compute_mds_error` function which is the next most intensive part of our program:

- Initial implementation  
```python
def compute_mds_error(D_goal,D_current,i=None):
    if i is None:
        return sum(sum(np.power(D_goal-D_current,2)))
    elif i<0 :
        return np.array([sum(np.power(D_goal[k]-D_current[k],2)) for k in range(len(D_goal))])
    else:
        return sum(np.power(D_goal[i]-D_current[i],2))
```

- Vectorized implementation
```python
def compute_mds_error(d_goal, d_current):
    return SUM((d_goal - d_current) ** 2)
```

We can see that the vectorized versions are cleaner and easier to understand. By running the benchmark again we can see it takes about 6 seconds, which is a 15x performance boost by simply vectorizing 2 functions. The flame graph for the vectorized version is shown below:

![Alt text](./vectorized.svg)

## Optimizing slow functions with Cython

