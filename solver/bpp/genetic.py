
#------------------------

import random
import itertools
import os
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass
class Box:
    id: int
    width: int
    height: int
    depth: int
    can_rotate: bool = True

    def get_orientations(self) -> List[Tuple[int, int, int]]:
        dims = [self.width, self.height, self.depth]
        if not self.can_rotate:
            return [tuple(dims)]
        orientations = list(set(itertools.permutations(dims, 3)))
        return orientations


@dataclass
class PlacedBox:
    box: Box
    x: int
    y: int
    z: int
    w: int
    h: int
    d: int


class Bin:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.placed_boxes: List[PlacedBox] = []

    def fits(self, w, h, d, x, y, z) -> bool:
    # Boundary checking
        if x + w > self.width or y + h > self.height or z + d > self.depth:
            return False

    # Overlap checking
        for pb in self.placed_boxes:
            if not (x + w <= pb.x or x >= pb.x + pb.w or
                y + h <= pb.y or y >= pb.y + pb.h or
                z + d <= pb.z or z >= pb.z + pb.d):
                return False

    # Gravitational support checking
        if z == 0:
            supported = True  # Placed directly on the ground
        else:
            supported = False
            for pb in self.placed_boxes:
            # Supported from below
                if (pb.x < x + w and pb.x + pb.w > x and
                    pb.y < y + h and pb.y + pb.h > y and
                    pb.z + pb.d == z):
                    supported = True
                    break
            if not supported:
                return False

        return True

    def get_candidate_positions(self) -> List[Tuple[int, int, int]]:
        positions = []  

        sorted_placed_boxes = self.placed_boxes

        for pb in sorted_placed_boxes:
            if pb.z == 0:  
                positions.append((pb.x + pb.w, pb.y, pb.z))  
                positions.append((pb.x, pb.y + pb.h, pb.z))  
                positions.append((pb.x, pb.y, pb.z + pb.d))  
            else:
                for potential_support in sorted_placed_boxes:
                    if (potential_support.x < pb.x + pb.w and potential_support.x + potential_support.w > pb.x and
                        potential_support.y < pb.y + pb.h and potential_support.y + potential_support.h > pb.y and
                        potential_support.z + potential_support.d == pb.z):  
                        positions.append((pb.x + pb.w, pb.y, pb.z))  
                        positions.append((pb.x, pb.y + pb.h, pb.z))  
                        positions.append((pb.x, pb.y, pb.z + pb.d))  
                        break  
        if not positions:
            positions.append((0, 0, 0))  
        positions.sort(key=lambda pos: (pos[2], pos[0]))  # First sort by ascending z-value; if z-values are equal, then sort by ascending x-value.
        return positions

    def place_box(self, box: Box) -> bool:
        """
        Try placing the box in all possible rotation orientations.
        """
        for orientation in box.get_orientations():
            w, h, d = orientation
            for x, y, z in self.get_candidate_positions():
                if self.fits(w, h, d, x, y, z):
                    self.placed_boxes.append(PlacedBox(box, x, y, z, w, h, d))
                    return True
        return False

def evaluate_individual(individual: List[int], boxes: List[Box], bin_size: Tuple[int, int, int], is_online: bool) -> int:
    bins = []  
    if is_online:
        # online
        for idx in individual:
            box = boxes[idx]
            placed = False
            for b in bins:
                if b.place_box(box):
                    placed = True
                    break
            if not placed:
                # create new bin
                new_bin = Bin(*bin_size)
                new_bin.place_box(box)
                bins.append(new_bin)
    else:
        # offline
        # Sort the boxes based on their dimensions or other rules.
        sorted_individual = sorted(individual, key=lambda idx: (boxes[idx].width * boxes[idx].height * boxes[idx].depth), reverse=True)
#        print(sorted_individual)
        for idx in sorted_individual:
            box = boxes[idx]
            placed = False
            for b in bins:
                if b.place_box(box):
                    placed = True
                    break
            if not placed:
                # create new bin
                new_bin = Bin(*bin_size)
                new_bin.place_box(box)
                bins.append(new_bin)
                
    return len(bins)

def create_individual(boxes: List[Box]) -> List[int]:
    indices = list(range(len(boxes)))
    random.shuffle(indices)
    return indices

def mutate(individual: List[int]):
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]

def crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    size = len(parent1)
    point = random.randint(1, size - 1)
    child_indices = parent1[:point]
    rest = [idx for idx in parent2 if idx not in child_indices]
    return child_indices + rest

def genetic_algorithm(box_sizes: List[List[int]], bin_size: List[int],
                      is_online: int, can_rotate: int,
                      population_size=30, generations=100, mutation_rate=0.2) -> Tuple[List[Bin], List[int]]:
    boxes = [Box(i, w, h, d, can_rotate=bool(can_rotate)) for i, (w, h, d) in enumerate(box_sizes)]
    if is_online:
        population = [list(range(len(boxes)))]  
    else:
        population = [create_individual(boxes) for _ in range(population_size)]
    for gen in range(generations):
        if is_online:
            population = sorted(population, key=lambda ind: evaluate_individual(ind, boxes, tuple(bin_size), is_online))
        else:
            population = sorted(population, key=lambda ind: evaluate_individual(ind, boxes, tuple(bin_size), is_online))
            next_gen = population[:5]  
            while len(next_gen) < population_size:
                if random.random() < 0.7:
                    p1, p2 = random.sample(population[:15], 2)
                    child = crossover(p1, p2)  
                else:
                    child = create_individual(boxes)  
                if random.random() < mutation_rate:
                    mutate(child)  
                next_gen.append(child)
            population = next_gen  
        best_score = evaluate_individual(population[0], boxes, tuple(bin_size), is_online)
    best_individual = population[0]
    bins = []
    for idx in best_individual:
        box = boxes[idx]
        placed = False
        for b in bins:
            if b.place_box(box):
                placed = True
                break
        if not placed:
            new_bin = Bin(*bin_size)
            new_bin.place_box(box)
            bins.append(new_bin)
    return bins, best_individual
