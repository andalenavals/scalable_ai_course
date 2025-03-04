#!/usr/bin/env python3

################################################################# 
#                                                               # 
#  This file has been written as a sample solution to an        # 
#  exercise in a course given at the High Performance           # 
#  Computing Centre Stuttgart (HLRS).                           # 
#  The examples are based on the examples in the MPI course of  # 
#  the Edinburgh Parallel Computing Centre (EPCC).              # 
#  It is made freely available with the understanding that      # 
#  every copy of this file must include this header and that    # 
#  HLRS and EPCC take no responsibility for the use of the      # 
#  enclosed teaching material.                                  # 
#                                                               # 
#  Authors: Joel Malard, Alan Simpson, (EPCC)                   # 
#           Rolf Rabenseifner, Tobias Haas (HLRS)               # 
#                                                               # 
#  Contact: rabenseifner@hlrs.de                                # 
#                                                               # 
#  Purpose: A first MPI example for this course                 # 
#                                                               # 
#  Contents: Python code, object send version (comm.send)       # 
#                                                               # 
#################################################################

# Note that you manually have to install mpi4py
from mpi4py import MPI

# application-related data
n = None
result = None

comm_world = MPI.COMM_WORLD
# MPI-related data
my_rank = comm_world.Get_rank() # or my_rank = MPI.COMM_WORLD.Get_rank()
num_procs = comm_world.Get_size() # or ditto ...

if (my_rank == 0):
   # reading the application data "n" from stdin only by process 0:
   n = int(input("Enter the number of elements (n): "))

# broadcasting the content of variable "n" in process 0 
# into variables "n" in all other processes:
n = comm_world.bcast(n, root=0)

# doing some application work in each process, e.g.:
result = 1.0 * my_rank * n
print(f"I am process {my_rank} out of {num_procs} handling the {my_rank}ith part of n={n} elements, result={result}")

if (my_rank != 0):
   # sending some results from all processes (except 0) to process 0:
   comm_world.send(result, dest=0, tag=99)
else:
   # receiving all these messages and, e.g., printing them 
   rank = None
   print(f"I'm proc 0: My own result is {result}") 
   for rank in range(1,num_procs):
      result = comm_world.recv(source=rank, tag=99)
      print(f"I'm proc 0: received result of process {rank} is {result}")
