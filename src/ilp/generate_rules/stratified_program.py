import itertools
from itertools import chain, combinations
import random

from src.ilp.dependency_graph import Dependency_Graph


class Stratified_Program:

    @staticmethod
    def generate_stratified_programs(clauses, max_program_size=20, verbose=False):
        '''
        Generates all stratified programs from a set of clauses.
        '''
        number_of_subsets_per_size = 1000
        all_programs = Stratified_Program.generate_subsets(clauses, max_program_size, number_of_subsets_per_size)
        stratified_programs = []
        total_programs = 0
        stratified_count = 0
        for program in all_programs:
            total_programs += 1
            dependency_graph = Dependency_Graph(program)
            if dependency_graph.is_stratified():
                stratified_programs.append([program])
                stratified_count += 1
        if verbose:
            print(f'Max program Size: {max_program_size}')
            print(f'Number of programs of each size: {number_of_subsets_per_size}')
            print(f'Total number of programs: {total_programs}')
            print(f'Stratified programs: {stratified_count}')
        return stratified_programs

    @staticmethod
    def random_subsets(sequence, number_of_subsets):
        '''
        Generate a set of random subsets.
        :param sequence: The set of which the subsets are generated from.
        :param number_of_subsets: Number of subsets from the set you want.
        :return: List of random subsets
        '''
        size = 2**len(sequence)
        power_set = Stratified_Program.power_set(sequence)
        random_subsets = []
        while number_of_subsets:
            print(number_of_subsets)
            index = random.randrange(size)
            subset = power_set[index]
            power_set[index] = power_set[size-1]
            number_of_subsets = number_of_subsets - 1
            size = size - 1
            random_subsets.append(subset)
        return random_subsets

    @staticmethod
    def power_set(sequence):
        '''
        Constructs power set of sequence
        '''
        s = list(sequence)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    @staticmethod
    def generate_subsets(iterable, max_subset_size, n_subsets_of_size):
        '''
        Generate a set of subsets.
        :param iterable: Set to create subsets from
        :param max_subset_size: Max size of the subsets
        :param n_subsets_of_size: Number of subsets of size k
        :return: list of lists. All subsets to the specifications.
        '''
        all_subsets = []
        for subset_size in range(max_subset_size):
            subsets = Stratified_Program.combinations(iterable, subset_size, n_subsets_of_size)
            all_subsets += subsets
        return all_subsets

    @staticmethod
    def combinations(iterable, r, p=None):  # !!
        n = len(iterable)
        if r > n:
            return
        indices = list(range(r))
        yield list(iterable[i] for i in indices)
        count = 1  # !!
        while p is None or count < p:  # !!
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield list(iterable[i] for i in indices)
            count += 1  # !!
        return iterable
