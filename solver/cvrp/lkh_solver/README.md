# **LKH-3**

LKH-3 is an extension of [LKH-2](http://webhotel4.ruc.dk/~keld/research/LKH) for solving constrained traveling salesman and vehicle routing problems. The extension has been desribed in the report

> K. Helsgaun,
> [*An Extension of the Lin-Kernighan-Helsgaun TSP Solver for Constrained Traveling Salesman and Vehicle Routing Problems*](http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3_REPORT.pdf).
> Technical Report, Roskilde University, 2017.

### Problem Types

Currently, LKH-3 is able to solve the following problem types:

From LKH-2:

- TSP: Symmetric traveling salesman problem
- ATSP: Asymmetric traveling salesman problem
- HCP: Hamiltonian cycle problem
- HPP: Hamiltonian path problem

New in LKH-3:

- [ACVRP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/ACVRP.tgz): Asymmetric capacitated vehicle routing problem
- [ADCVRP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/ADCVRP.tgz): Asymmetric distance constrained vehicle routing problem
- [BMTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/BMTSP.tgz): Bounded multiple traveling salesman problem
- [BWTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/BWTSP.tgz): Black and white traveling salesman problem
- [CATSPP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/CATSPP.tgz): Constrained asymmetric traveling salesman path problem
- [CBTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/CBTSP.tgz): Colored balanced traveling salesman problem
- [CBnTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/CBnTSP.tgz): Colored bottleneck traveling salesman problem
- [CluVRP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/CluVRP.tgz): Clustered vehicle routing problem
- [CCCTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/CCCTSP.tgz): Cumulative capacitated colored traveling salesman problem
- [CCVRP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/CCVRP.tgz): Cumulative capacitated vehicle routing problem
- [CTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/CTSP.tgz): Colored traveling salesman problem
- [CTSP-d](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/CTSP-d.tgz): Clustered traveling salesman problem with *d*-relaxed priority rule
- [CVRP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/CVRP.tgz): Capacitated vehicle routing problem
- [CVRPTW](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/CVRPTW.tgz): Capacitated vehicle routing problem with time windows
- [DCVRP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/DCVRP.tgz): Distance constrained capacitated vehicle routing problem
- [GCTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/GCTSP.tgz): General colored traveling salesmen problem
- [k-TSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/k-TSP.tgz): K-traveling salesman problem
- [1-PDTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/1-PDTSP.tgz): One-commodity pickup-and-delivery traveling salesman problem
- [m-PDTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/m-PDTSP.tgz): Multi-commodity pickup-and-delivery traveling salesman problem
- [m1-PDTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/m1-PDTSP.tgz): Multi-commodity one-to-one pickup-and-delivery traveling salesman problem
- [MDMTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/MDMTSP.tgz): Multiple depot multiple traveling salesman problem
- [MLP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/MLP.tgz): Minimum latency problem
- [MSCTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/MSCTSP.tgz): Maximum scattered colored traveling salesman problem
- [MTRP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/MTRP.tgz): Multiple traveling repairman problem
- [MTRPD](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/MTRPD.tgz): Multiple traveling repairman problem with distance constraints
- [mTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/mTSP.tgz): Multiple traveling salesmen problem
- [OCMTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/OCMTSP.tgz): Open close multiple traveling salesman problem
- [OVRP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/OVRP.tgz): Open vehicle routing problem
- [PCTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/PCTSP.tgz): Precedence-constrained colored traveling salesman problem
- [PDPTW](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/PDPTW.tgz): Pickup-and-delivery problem with time windows
- [PDTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/PDTSP.tgz): Pickup-and-delivery traveling salesman problem
- [PDTSPF](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/PDTSPF.tgz): Pickup-and-delivery traveling salesman problem with FIFO loading
- [PDTSPL](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/PDTSPL.tgz): Pickup-and-delivery traveling salesman problem with LIFO loading
- [PTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/PTSP.tgz): Homogeneous probability traveling salesman problem
- [RCTVRP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/RCTVRP.tgz): Risk-constrained cash-in-transit vehicle routing problem
- [RCTVRPTW](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/RCTVRPTW.tgz): Risk-constrained cash-in-transit vehicle routing with time windows
- [SoftCluVRP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/SoftCluVRP.tgz): Soft-clustered vehicle routing problem
- [SOP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/SOP.tgz): Sequential ordering problem
- [STTSP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/STTSP.tgz): Steiner traveling salesman problem
- [TRP](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/TRP.tgz): Traveling repairman problem
- [TSPDL](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/TSPDL.tgz): Traveling salesman problem with draft limits
- [TSPPD](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/TSPPD.tgz): Traveling salesman problem with pickups and deliveries
- [TSPTW](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/TSPTW.tgz): Traveling salesman problem with time windows
- [VRPB](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/VRPB.tgz): Vehicle routing problem with backhauls
- [VRPBTW](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/VRPBTW.tgz): Vehicle routing problem with backhauls and time windows
- [VRPMPD](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/VRPMPD.tgz): Vehicle routing problem with mixed pickup and delivery
- [VRPMPDTW](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/VRPMPDTW.tgz): Vehicle routing problem with mixed pickup and delivery and time windows
- [VRPSPD](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/VRPSPD.tgz): Vehicle routing problem with simultaneous pickup and delivery
- [VRPSPDTW](http://webhotel4.ruc.dk/~keld/research/LKH-3/BENCHMARKS/VRPSPDTW.tgz): Vehicle routing problem with simultaneous pickup-delivery and time windows

Extensive testing on benchmark instances from the literature has shown that LKH-3 is effective. Best known solutions are often obtained, and in some cases, new best solutions are found. Instances and best solutions obtained by LKH-3 may be downloaded by clicking on the problem types above. Unpack a downloaded file, *file_name.*tgz*,* by executing

```bash
tar xvfz *file_name*.tgz
```

Run scripts are provided for Unix/Linux.

A list of scientific applications of LKH may be seen [here](http://webhotel4.ruc.dk/~keld/research/LKH/ScientificApplications.html).

### Installation

LKH-3 has been implemented in the programming language C. The software is entirely written in ANSI C and portable across a number of computer platforms and C compilers.

The code can be downloaded here: [LKH-3.0.13.tgz](http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz).

On a Unix/Linux machine execute the following commands:

```bash
tar xvfz LKH-3.0.13.tgz
cd LKH-3.0.13
make
```

An executable file called LKH will now be available in the directory LKH-3.0.13.