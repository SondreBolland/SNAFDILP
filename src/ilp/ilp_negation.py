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
        self.uses_negation = self.uses_negation()
        if self.uses_negation:
            self.extend_examples()

    def uses_negation(self):
        '''
        Checks if any of the rule templates allow the use of negation
        :return: true if negation is allowed, false if not
        '''
        rules = self.program_template.rules
        for rule_template_pair in rules.values():
            for rule_template in rule_template_pair:
                if rule_template is None:
                    continue
                allow_negation = rule_template.neg
                if allow_negation:
                    return True
        return False

    def extend_examples(self):
        '''
        For every positive and negative example we construct its negation
        and add it to the set of examples.
        E.g.: P = {zero(0),...} then we add "not zero(0)"
        to the negative examples
        '''
        negative_literals_positive_examples = []
        negative_literals_negative_examples = []
        for pos in self.positive:
            negative_literal = pos.__copy__()
            negative_literal.negate()
            negative_literals_negative_examples.append(negative_literal)
        for neg in self.negative:
            negative_literal = neg.__copy__()
            negative_literal.negate()
            negative_literals_positive_examples.append(negative_literal)

        self.positive += negative_literals_positive_examples
        self.negative += negative_literals_negative_examples

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
        # Build ground, non negated, atoms
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
        if self.uses_negation:
            for literal in ground_positive_literals:
                #if atom == Atom([],'⊥'):
                #    continue
                negative_literal = literal.__copy__()
                negative_literal.negate()
                ground_literals.append(negative_literal)
        return ground_literals

    def convert(self):
        '''Generate initial valuations
        '''
        ground_literals = self.generate_ground_literals()
        valuation_mapping = {}
        initial_valuation = []
        predicate_valuation_idx_map = {}

        current_predicate = ground_literals[0].predicate
        start_idx = 0
        n_positive_literals = (len(ground_literals)/2) if self.uses_negation else len(ground_literals)
        for idx, literal in enumerate(ground_literals):
            predicate = literal.predicate
            if predicate != current_predicate and idx < n_positive_literals:
                predicate_slice = [start_idx, idx-1]
                predicate_valuation_idx_map.update({current_predicate: predicate_slice})
                start_idx = idx
                current_predicate = predicate
            elif idx+1 == n_positive_literals:
                predicate_slice = [start_idx, idx]
                predicate_valuation_idx_map.update({current_predicate: predicate_slice})

            if literal.negated:
                # Create positive version of the literal to get it's valuation
                positive_version = literal.__copy__()
                positive_version.negate()
                positive_version_valuation = initial_valuation[valuation_mapping[positive_version]]
                complement_valuation = 1 - positive_version_valuation
                initial_valuation.append(complement_valuation)
                valuation_mapping[literal] = idx
            elif literal in self.background:
                initial_valuation.append(1)
                valuation_mapping[literal] = idx
            else:
                initial_valuation.append(0)
                valuation_mapping[literal] = idx

        print(predicate_valuation_idx_map)
        return (np.array(initial_valuation, dtype=np.float32), valuation_mapping, predicate_valuation_idx_map)
