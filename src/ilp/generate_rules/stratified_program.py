import itertools
from itertools import chain, combinations
import random
from collections import defaultdict

from src.core import Clause
from src.ilp.dependency_graph import Dependency_Graph


class Stratified_Program:

    def __init__(self, program: list):
        dependency_graph = Dependency_Graph(program)
        if not dependency_graph.is_stratified():
            raise ValueError("Program must be stratified.")
        self.program = program

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
                stratified_programs.append(program)
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

    def partition_program(self):
        clauses = self.program

        predicate_list = defaultdict(list)
        for clause in clauses:
            if clause is None:
                continue
            predicate = clause.head.predicate
            predicate_list[predicate].append(clause)

        # Create first stratum p_0
        p_0 = []
        defined_predicate = []
        for predicate in predicate_list.keys():
            is_definite = True
            for clause in predicate_list[predicate]:
                if self.has_negation(clause):
                    is_definite = False
            if is_definite:
                p_0.extend(predicate_list[predicate])
                defined_predicate.append(predicate)
        # Remove all p_0 predicates from program
        for predicate in defined_predicate:
            predicate_list.pop(predicate)

        # Add all remaining strata
        strata = defaultdict(list)
        strata[0] = p_0
        stratum_index = 1
        while len(predicate_list) != 0:
            predicate_list, current_stratum = self.next_stratum(predicate_list, strata)
            strata[stratum_index].extend(current_stratum)
            stratum_index += 1
        return strata

    def next_stratum(self, predicate_list, strata):
        stratum = []
        defined_predicate = []
        for predicate in predicate_list.keys():
            defined_negated_body = True
            for clause in predicate_list[predicate]:
                for literal in clause.body:
                    if literal.negated:
                        name = literal.predicate
                        if not self.is_defined(name, strata):
                            defined_negated_body = False
                if not defined_negated_body:
                    break
            if not defined_negated_body:
                stratum.extend(predicate_list[predicate])
                defined_predicate.append(predicate)

        for predicate in defined_predicate:
            predicate_list.pop(predicate)

        return predicate_list, stratum

    def is_defined(self, predicate, strata):
        '''
        Check if given predicate is defined in the list of strata
        :param predicate: string
        :param strata: list of list of clauses
        :return: true if predicate is defined, false if not
        '''
        for stratum in strata.keys():
            for clause in strata[stratum]:
                head = clause.head
                if head.predicate == predicate:
                    return True
        return False

    def get_definite_clauses(self, clauses):
        '''
        Finds all definite clauses in set of clauses
        :param clauses:
        :return: list of all definite clauses and
         list of all other clauses
        '''
        definite_clauses = []
        rest_clauses = []
        for clause in clauses:
            if not self.has_negation(clause):
                definite_clauses.append(clause)
            else:
                rest_clauses.append(clause)
        return definite_clauses, rest_clauses

    def has_negation(self, clause):
        '''
        Checks if the clause has a literal in the body that is negated
        :param clause: Logic program clause
        :return: true if negated literal in body, false if not
        '''
        if type(clause) != Clause:
            raise ValueError("Argument must be of type Clause")
        for literal in clause.body:
            if literal.negated:
                return True
        return False

    def get_predicates(self, clauses):
        predicates = []
        for clause in clauses:
            if clause is None:
                continue
            head = clause.head
            predicates.append(head.predicate)
        return predicates
