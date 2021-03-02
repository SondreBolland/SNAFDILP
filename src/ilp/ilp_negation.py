'''Defines the ILP problem
'''

from src.ilp import Language_Frame, Program_Template
from src.core import Atom, Term, Literal
import numpy as np


class ILP_Negation():

    def __init__(self, language_frame: Language_Frame, background: list, positive: list, negative: list, program_template: Program_Template):
        '''
        Arguments:
            language_frame {Language_Frame} -- language frame
            background {list} -- background assumptions
            positive {list} -- positive examples
            negative {list} -- negative examples
            program_template {Program_Template} -- program template
        '''
        self.language_frame = language_frame
        self.background = background
        self.positive = positive
        self.negative = negative
        self.program_template = program_template

    def generate_ground_literals(self):
        '''Generates the ground atoms from p_i,p_a,target and constants
        '''
        p = list(set(self.language_frame.p_e +
                     self.program_template.p_a + [self.language_frame.target]))
        constants = self.language_frame.constants

        # Build constant matrix
        constant_matrix = []
        for const1 in constants:
            for const2 in constants:
                term1 = Term(False, const1)
                term2 = Term(False, const2)
                constant_matrix.append([term1, term2])
        # Build ground atoms
        ground_positive_literals = []
        ground_positive_literals.append(Literal(Atom([], '⊥'), False))
        added_literals = {}
        for pred in p:
            for term in constant_matrix:
                literal = Literal(Atom([term[i]
                             for i in range(0, pred.arity)], pred.predicate), False)
                if literal not in added_literals:
                    ground_positive_literals.append(literal)
                    added_literals[literal] = 1
        ground_literals = []
        ground_literals += ground_positive_literals
        for atom in ground_positive_literals:
            if atom == Atom([],'⊥'):
                continue
            negative_literal = Literal(atom, True)
            ground_literals.append(negative_literal)
        return ground_literals

    def convert(self):
        '''Generate initial valuations
        '''
        ground_literals = self.generate_ground_literals()
        valuation_mapping = {}
        initial_valuation = []
        for idx, literal in enumerate(ground_literals):
            if literal.negated:
                # Create positive version of the literal to get it's valuation
                positive_version = literal.__copy__()
                positive_version.negate()
                positive_version_valuation = valuation_mapping[positive_version]
                complement_valuation = 1 - positive_version_valuation
                initial_valuation.append(complement_valuation)
                valuation_mapping[literal] = idx
            elif literal in self.background:
                initial_valuation.append(1)
                valuation_mapping[literal] = idx
            else:
                initial_valuation.append(0)
                valuation_mapping[literal] = idx
        return (np.array(initial_valuation, dtype=np.float32), valuation_mapping)
