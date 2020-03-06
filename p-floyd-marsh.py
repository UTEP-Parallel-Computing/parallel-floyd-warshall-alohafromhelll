from mpi4py import MPI
import timeit
import math

#read matrix from file (fwTest.txt)
with open('fwTest.txt', 'r') as f:
    matrix = [[int(x) for x in line.split()] for line in f]

# open file for writing
out_file = open('results.txt', 'w')

def fw(graph):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rowsPerThread = len(graph) / size
    threadsPerRow = size / len(graph)
    startingRow = math.trunc(rowsPerThread * rank)
    endingRow = math.trunc(rowsPerThread * (rank + 1))

    for k in range(len(graph)):
        ownerofK = (threadsPerRow * k)
        graph[k] = comm.bcast(graph[k], root = ownerofK)
        for x in range(startingRow, endingRow):
            for y in range(len(graph)):
                graph[x][y] = min(graph[x][y], graph[x][k] + graph[k][y])

    if rank == 0:
        for k in range(endingRow, len(graph)):
            graph[k] - comm.recv(source = ownerofK, tag = k)
    else:
        for k in range(startingRow, endingRow):
            comm.send(graph[k], dest = 0, tag = k)

#start timer
start = timeit.default_timer()

#call function
fw(matrix)
print(matrix)

#stop timer
stop = timeit.default_timer()
#print time
print('Time: ', stop - start)

#write to a file called results.txt
print (matrix, file = out_file)
