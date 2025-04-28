from bpp_solver import genetic_algorithm
import os
import random

random.seed(20)
def generate_random_list(n):
    return [[random.randint(1, 10) for _ in range(3)] for _ in range(n)]
dimension="2D"       # input:"2D" or "3D"
is_online=0          # input:0 or 1
can_rotate=0         # inut:1 or 0
bin_size =[10,10]           # input:bin_size--[10,10] or [10, 10,10]
box_sizes= generate_random_list(20)          # input:item_size--
if dimension=="2D":
    bin_size=[10,1,10]
    box_sizes=[[row[0], 1, row[1]] for row in box_sizes]

try:
        bins, best_individual = genetic_algorithm(box_sizes, bin_size, is_online, can_rotate)
        print("Problem solved successfully!Number of bins used:",len(bins))
except Exception:
        print(False) 

