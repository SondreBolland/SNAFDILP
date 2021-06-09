'''For testing DILP with negation
'''
import itertools
import random

from src.core import Term, Atom, Literal
from src.ilp import Language_Frame, Program_Template, Rule_Template
from src.dilp import DILP
import tensorflow as tf

from src.ilp.template.rule_template_negation import Rule_Template_Negation

tf.compat.v1.enable_eager_execution()

def no_negative_cycle():
    '''
    Learn the target predicate negative_cycle(X). The predicate checks if
    a given node in a graph is part of a negative cycle (cycle with at least one
    negative edge).
    '''
    constants = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    edge_relations = [Atom([Term(False, 'a'), Term(False, 'b')], 'edge'),
                     Atom([Term(False, 'a'), Term(False, 'c')], 'edge'),
                     Atom([Term(False, 'b'), Term(False, 'c')], 'edge'),
                     Atom([Term(False, 'c'), Term(False, 'd')], 'edge'),
                     Atom([Term(False, 'c'), Term(False, 'e')], 'edge'),
                     Atom([Term(False, 'd'), Term(False, 'h')], 'edge'),
              Atom([Term(False, 'h'), Term(False, 'd')], 'edge'),
                     Atom([Term(False, 'f'), Term(False, 'a')], 'edge'),
                     Atom([Term(False, 'f'), Term(False, 'b')], 'edge'),
                     Atom([Term(False, 'g'), Term(False, 'e')], 'edge'),
                     Atom([Term(False, 'g'), Term(False, 'c')], 'edge'),
                     Atom([Term(False, 'e'), Term(False, 'a')], 'edge')]
    edge_type = [Atom([Term(False, 'a'), Term(False, 'b')], 'positive'),
                 Atom([Term(False, 'a'), Term(False, 'c')], 'negative'),
                 Atom([Term(False, 'b'), Term(False, 'c')], 'positive'),
                 Atom([Term(False, 'c'), Term(False, 'd')], 'positive'),
                 Atom([Term(False, 'c'), Term(False, 'e')], 'negative'),
                 Atom([Term(False, 'd'), Term(False, 'e')], 'positive'),
                 Atom([Term(False, 'h'), Term(False, 'd')], 'positive'),
                 Atom([Term(False, 'f'), Term(False, 'a')], 'positive'),
                 Atom([Term(False, 'f'), Term(False, 'b')], 'positive'),
                 Atom([Term(False, 'g'), Term(False, 'e')], 'positive'),
                 Atom([Term(False, 'g'), Term(False, 'c')], 'positive'),
                 Atom([Term(False, 'e'), Term(False, 'a')], 'positive')]
    B_atom = edge_relations + edge_type
    B = [Literal(atom, False) for atom in B_atom]

    P_atom = [Atom([Term(False, 'f')], 'target'),
              Atom([Term(False, 'g')], 'target'),
              Atom([Term(False, 'd')], 'target'),
              Atom([Term(False, 'h')], 'target')]
    P = [Literal(atom, False) for atom in P_atom]

    N_atom = [Atom([Term(False, 'a')], 'target'),
              Atom([Term(False, 'b')], 'target'),
              Atom([Term(False, 'c')], 'target'),
              Atom([Term(False, 'e')], 'target')]
    N = [Literal(atom, False) for atom in N_atom]

    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')
    term_x_2 = Term(True, 'X_2')

    p_e = [Literal(Atom([term_x_0, term_x_1], 'edge'), False),
           Literal(Atom([term_x_0, term_x_1], 'positive'), False),
           Literal(Atom([term_x_0, term_x_1], 'negative'), False)]
    p_a = [Literal(Atom([term_x_0, term_x_2], 'pred'), False)]
    target = Literal(Atom([term_x_0], 'target'), False)

    # Define rules for intensional predicates
    p_a_rule = [(Rule_Template_Negation(0, False, False), Rule_Template_Negation(1, True, False))]
    target_rule = (Rule_Template_Negation(0, True, True),
                   None)
    rules = {p_a[0]: p_a_rule[0], target: target_rule}

    language_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    # program_template = Program_Template(p_a, rules, 300)

    dilp = DILP(language_frame, B, P, N, program_template)
    return dilp.train(steps=300)

def not_grandparent():
    '''
    Learn the target predicate not_grandparent(X,Y) which is true
    if X is not the grapndparent of Y.
    '''
    constants = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    B_atom = [Atom([Term(False, 'a'), Term(False, 'b')], 'mother'),
              Atom([Term(False, 'i'), Term(False, 'a')], 'father'),
              Atom([Term(False, 'a'), Term(False, 'b')], 'father'),
              Atom([Term(False, 'a'), Term(False, 'c')], 'father'),
              Atom([Term(False, 'b'), Term(False, 'd')], 'father'),
              Atom([Term(False, 'b'), Term(False, 'e')], 'mother'),
              Atom([Term(False, 'c'), Term(False, 'f')], 'mother'),
              Atom([Term(False, 'c'), Term(False, 'g')], 'mother'),
              Atom([Term(False, 'f'), Term(False, 'h')], 'mother')]
    B = [Literal(atom, False) for atom in B_atom]

    N_atom = [Atom([Term(False, 'i'), Term(False, 'b')], 'target'),
              Atom([Term(False, 'i'), Term(False, 'c')], 'target'),
              Atom([Term(False, 'a'), Term(False, 'd')], 'target'),
              Atom([Term(False, 'a'), Term(False, 'e')], 'target'),
              Atom([Term(False, 'a'), Term(False, 'f')], 'target'),
              Atom([Term(False, 'a'), Term(False, 'g')], 'target'),
              Atom([Term(False, 'c'), Term(False, 'h')], 'target')]

    possible_target_atoms = [Atom([Term(False, x), Term(False, y)], 'target') for x, y in
                             itertools.product(constants, constants)]
    P_atom = []
    for atom in possible_target_atoms:
        if atom not in N_atom:
            c1 = atom.terms[0].name
            c2 = atom.terms[1].name
            P_atom.append(Atom([Term(False, c1), Term(False, c2)], 'target'))
    P_atom = random.sample(P_atom, len(N_atom))

    P = [Literal(atom, False) for atom in P_atom]
    N = [Literal(atom, False) for atom in N_atom]

    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')
    term_x_2 = Term(True, 'X_2')

    p_e = [Literal(Atom([term_x_0, term_x_1], 'mother'), False),
           Literal(Atom([term_x_0, term_x_1], 'father'), False)]
    p_a = [Literal(Atom([term_x_0, term_x_1], 'pred1'), False),
           Literal(Atom([term_x_0, term_x_1], 'pred2'), False)]
    target = Literal(Atom([term_x_0, term_x_2], 'target'), False)

    # Define rules for intensional predicates
    p_a_rule = [(Rule_Template_Negation(1, True, False), None),
                (Rule_Template_Negation(0, False, False), Rule_Template_Negation(0, False, False))]
    target_rule = (Rule_Template_Negation(0, True, True),
                   None)
    rules = {p_a[0]: p_a_rule[0], p_a[1]: p_a_rule[1], target: target_rule}

    language_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    # program_template = Program_Template(p_a, rules, 300)

    dilp = DILP(language_frame, B, P, N, program_template)
    return dilp.train(steps=300)


def two_children():
    '''
    Learn the target predicate of has_at_least_two_children(X).
    In a directed graph we want to check if a node has at least two children
    (that are not equal).
    '''
    constants = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    edge_relation = [Atom([Term(False, 'a'), Term(False, 'b')], 'edge'),
                     Atom([Term(False, 'a'), Term(False, 'c')], 'edge'),
                     Atom([Term(False, 'b'), Term(False, 'd')], 'edge'),
                     Atom([Term(False, 'c'), Term(False, 'd')], 'edge'),
                     Atom([Term(False, 'c'), Term(False, 'e')], 'edge'),
                     Atom([Term(False, 'd'), Term(False, 'e')], 'edge'),
                     Atom([Term(False, 'f'), Term(False, 'a')], 'edge'),
                     Atom([Term(False, 'f'), Term(False, 'b')], 'edge'),
                     Atom([Term(False, 'g'), Term(False, 'e')], 'edge'),
                     Atom([Term(False, 'g'), Term(False, 'c')], 'edge')]
    equals_relation = [Atom([Term(False, x), Term(False, y)], 'equals') for x, y in zip(constants, constants) if x == y]
    B_atom = edge_relation + equals_relation
    B = [Literal(atom, False) for atom in B_atom]

    P_atom = [Atom([Term(False, 'a')], 'target'),
              Atom([Term(False, 'c')], 'target'),
              Atom([Term(False, 'f')], 'target'),
              Atom([Term(False, 'g')], 'target')]
    P = [Literal(atom, False) for atom in P_atom]

    N_atom = [Atom([Term(False, 'b')], 'target'),
              Atom([Term(False, 'd')], 'target'),
              Atom([Term(False, 'e')], 'target'),
              Atom([Term(False, 'h')], 'target')]
    N = [Literal(atom, False) for atom in N_atom]

    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')
    term_x_2 = Term(True, 'X_2')

    p_e = [Literal(Atom([term_x_0, term_x_1], 'edge'), False),
           Literal(Atom([term_x_0, term_x_1], 'equals'), False)]
    p_a = [Literal(Atom([term_x_0, term_x_2], 'pred'), False)]
    target = Literal(Atom([term_x_0], 'target'), False)

    # Define rules for intensional predicates
    p_a_rule = (Rule_Template_Negation(1, False, True), None)
    target_rule = (Rule_Template_Negation(1, True, False),
                   None)
    rules = {p_a[0]: p_a_rule, target: target_rule}

    language_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    # program_template = Program_Template(p_a, rules, 300)

    dilp = DILP(language_frame, B, P, N, program_template)
    return dilp.train(steps=300)


def has_roommate():
    '''
    Learn the target predicate of has_roommate(X, Y)
    which is true if there exists a person Y that X is married to
    and Y is not a researcher.
    '''
    constants = ['Paul', 'Randy', 'Rachel', 'Bob', 'Alice', 'Steve', 'Frank', 'Julia',
                 'Stan', 'Kyle', 'Peter', 'Tony', 'Monica', 'Carl', 'Dolores', 'Tommy',
                 'Pedro', 'Will', 'Sophie', 'Eric', 'Jon', 'Robert', 'Sansa', 'Arya',
                 'Tormund', 'Mance']
    # married(X,Y) with symmetric closure
    B_atom = [Atom([Term(False, 'Bob'), Term(False, 'Rachel')], 'married'),
              Atom([Term(False, 'Rachel'), Term(False, 'Bob')], 'married'),
              Atom([Term(False, 'Randy'), Term(False, 'Alice')], 'married'),
              Atom([Term(False, 'Alice'), Term(False, 'Randy')], 'married'),
              Atom([Term(False, 'Steve'), Term(False, 'Julia')], 'married'),
              Atom([Term(False, 'Julia'), Term(False, 'Steve')], 'married'),
              Atom([Term(False, 'Frank'), Term(False, 'Monica')], 'married'),
              Atom([Term(False, 'Monica'), Term(False, 'Frank')], 'married'),
              Atom([Term(False, 'Stan'), Term(False, 'Dolores')], 'married'),
              Atom([Term(False, 'Dolores'), Term(False, 'Frank')], 'married'),
              Atom([Term(False, 'Will'), Term(False, 'Sophie')], 'married'),
              Atom([Term(False, 'Sophie'), Term(False, 'Will')], 'married'),
              Atom([Term(False, 'Eric'), Term(False, 'Arya')], 'married'),
              Atom([Term(False, 'Arya'), Term(False, 'Eric')], 'married'),
              # researcher(X)
              Atom([Term(False, 'Robert')], 'researcher'),
              Atom([Term(False, 'Eric')], 'researcher'),
              Atom([Term(False, 'Bob')], 'researcher'),
              Atom([Term(False, 'Frank')], 'researcher'),
              Atom([Term(False, 'Monica')], 'researcher'),
              Atom([Term(False, 'Mance')], 'researcher'),
              Atom([Term(False, 'Jon')], 'researcher'),
              Atom([Term(False, 'Tormund')], 'researcher'),
              Atom([Term(False, 'Sansa')], 'researcher')]
    B = [Literal(atom, False) for atom in B_atom]

    P_atom = [Atom([Term(False, 'Randy')], 'target'),
              Atom([Term(False, 'Alice')], 'target'),
              Atom([Term(False, 'Steve')], 'target'),
              Atom([Term(False, 'Julia')], 'target'),
              Atom([Term(False, 'Dolores')], 'target'),
              Atom([Term(False, 'Stan')], 'target'),
              Atom([Term(False, 'Will')], 'target'),
              Atom([Term(False, 'Sophie')], 'target')]
    P = [Literal(atom, False) for atom in P_atom]

    N_atom = [Atom([Term(False, 'Bob')], 'target'),
              Atom([Term(False, 'Rachel')], 'target'),
              Atom([Term(False, 'Frank')], 'target'),
              Atom([Term(False, 'Monica')], 'target'),
              Atom([Term(False, 'Kyle')], 'target'),
              Atom([Term(False, 'Peter')], 'target'),
              Atom([Term(False, 'Tony')], 'target'),
              Atom([Term(False, 'Carl')], 'target'),
              Atom([Term(False, 'Tommy')], 'target'),
              Atom([Term(False, 'Pedro')], 'target'),
              Atom([Term(False, 'Tommy')], 'target'),
              Atom([Term(False, 'Eric')], 'target'),
              Atom([Term(False, 'Arya')], 'target'),
              Atom([Term(False, 'Jon')], 'target'),
              Atom([Term(False, 'Robert')], 'target'),
              Atom([Term(False, 'Mance')], 'target'),
              Atom([Term(False, 'Tormund')], 'target')]
    N = [Literal(atom, False) for atom in N_atom]

    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')

    p_e = [Literal(Atom([term_x_0, term_x_1], 'married'), False),
           Literal(Atom([term_x_0], 'researcher'), False)]
    p_a = [Literal(Atom([term_x_0, term_x_1], 'pred'), False)]
    target = Literal(Atom([term_x_0], 'target'), False)

    # Define rules for intensional predicates
    p_a_rule = (Rule_Template_Negation(0, False, True), None)
    target_rule = (Rule_Template_Negation(1, True, False),
                   None)
    rules = {p_a[0]: p_a_rule, target: target_rule}

    language_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    # program_template = Program_Template(p_a, rules, 300)

    dilp = DILP(language_frame, B, P, N, program_template)
    return dilp.train(steps=200)


def orphan():
    '''
    Learn the target predicate of orphan(X)
    which is true if X has no parent
    '''
    constants = ['Paul', 'Randy', 'Rachel', 'Bob', 'Alice', 'Steve', 'Frank', 'Julia',
                 'Stan', 'Kyle', 'Peter', 'Tony', 'Monica', 'Carl', 'Dolores', 'Tommy',
                 'Pedro', 'Will', 'Sophie', 'Eric', 'Jon', 'Robert', 'Sansa', 'Arya',
                 'Tormund', 'Mance']
    # parent(X, Y) is true if X is the father of Y
    B_atom = [Atom([Term(False, 'Bob'), Term(False, 'Paul')], 'parent'),
              Atom([Term(False, 'Bob'), Term(False, 'Alice')], 'parent'),
              Atom([Term(False, 'Randy'), Term(False, 'Tony')], 'parent'),
              Atom([Term(False, 'Randy'), Term(False, 'Stan')], 'parent'),
              Atom([Term(False, 'Peter'), Term(False, 'Kyle')], 'parent'),
              Atom([Term(False, 'Alice'), Term(False, 'Kyle')], 'parent'),
              Atom([Term(False, 'Tony'), Term(False, 'Carl')], 'parent'),
              Atom([Term(False, 'Dolores'), Term(False, 'Tony')], 'parent'),
              Atom([Term(False, 'Stan'), Term(False, 'Julia')], 'parent'),
              Atom([Term(False, 'Julia'), Term(False, 'Tommy')], 'parent'),
              Atom([Term(False, 'Kyle'), Term(False, 'Tommy')], 'parent'),
              Atom([Term(False, 'Carl'), Term(False, 'Sophie')], 'parent'),
              Atom([Term(False, 'Carl'), Term(False, 'Eric')], 'parent'),
              Atom([Term(False, 'Tommy'), Term(False, 'Arya')], 'parent'),
              Atom([Term(False, 'Tommy'), Term(False, 'Tormund')], 'parent'),
              Atom([Term(False, 'Tommy'), Term(False, 'Mance')], 'parent')]
    B = [Literal(atom, False) for atom in B_atom]

    P_atom = [Atom([Term(False, 'Randy')], 'target'),
              Atom([Term(False, 'Bob')], 'target'),
              Atom([Term(False, 'Peter')], 'target'),
              Atom([Term(False, 'Rachel')], 'target'),
              Atom([Term(False, 'Monica')], 'target'),
              Atom([Term(False, 'Dolores')], 'target'),
              Atom([Term(False, 'Steve')], 'target'),
              Atom([Term(False, 'Frank')], 'target'),
              Atom([Term(False, 'Pedro')], 'target'),
              Atom([Term(False, 'Will')], 'target'),
              Atom([Term(False, 'Jon')], 'target'),
              Atom([Term(False, 'Robert')], 'target'),
              Atom([Term(False, 'Sansa')], 'target'), ]
    P = [Literal(atom, False) for atom in P_atom]

    N_atom = [Atom([Term(False, 'Paul')], 'target'),
              Atom([Term(False, 'Alice')], 'target'),
              Atom([Term(False, 'Kyle')], 'target'),
              Atom([Term(False, 'Tony')], 'target'),
              Atom([Term(False, 'Stan')], 'target'),
              Atom([Term(False, 'Carl')], 'target'),
              Atom([Term(False, 'Julia')], 'target'),
              Atom([Term(False, 'Tommy')], 'target'),
              Atom([Term(False, 'Sophie')], 'target'),
              Atom([Term(False, 'Eric')], 'target'),
              Atom([Term(False, 'Arya')], 'target'),
              Atom([Term(False, 'Tormund')], 'target'),
              Atom([Term(False, 'Mance')], 'target')]
    N = [Literal(atom, False) for atom in N_atom]

    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')

    p_e = [Literal(Atom([term_x_0, term_x_1], 'parent'), False)]
    p_a = []
    target = Literal(Atom([term_x_0], 'target'), False)

    # Define rules for intensional predicates
    p_a_rule = (None, None)
    target_rule = (Rule_Template_Negation(1, False, True),
                   None)
    rules = {target: target_rule}

    language_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    # program_template = Program_Template(p_a, rules, 300)

    dilp = DILP(language_frame, B, P, N, program_template)
    return dilp.train(steps=200)


def innocent():
    '''
    Learn the target predicate of innocent(X)
    which is true if X is not guilty
    '''
    constants = ['Paul', 'Randy', 'Rachel', 'Bob', 'Alice',
                 'Stan', 'Kyle', 'Peter', 'Tony', 'Monica']
    B_atom = [Atom([Term(False, 'Bob')], 'guilty'),
              Atom([Term(False, 'Randy')], 'guilty'),
              Atom([Term(False, 'Peter')], 'guilty'),
              Atom([Term(False, 'Alice')], 'guilty'),
              Atom([Term(False, 'Monica')], 'guilty')]
    B = [Literal(atom, False) for atom in B_atom]

    P_atom = [Atom([Term(False, 'Paul')], 'target'),
              Atom([Term(False, 'Rachel')], 'target'),
              Atom([Term(False, 'Kyle')], 'target'),
              Atom([Term(False, 'Tony')], 'target'),
              Atom([Term(False, 'Stan')], 'target')]
    P = [Literal(atom, False) for atom in P_atom]

    N_atom = [Atom([Term(False, 'Bob')], 'target'),
              Atom([Term(False, 'Randy')], 'target'),
              Atom([Term(False, 'Peter')], 'target'),
              Atom([Term(False, 'Alice')], 'target'),
              Atom([Term(False, 'Monica')], 'target')]
    N = [Literal(atom, False) for atom in N_atom]

    term_x_0 = Term(True, 'X_0')

    p_e = [Literal(Atom([term_x_0], 'guilty'), False)]
    p_a = []
    target = Literal(Atom([term_x_0], 'target'), False)

    # Define rules for intensional predicates
    p_a_rule = (None, None)
    target_rule = (Rule_Template_Negation(0, False, True),
                   None)
    rules = {target: target_rule}

    language_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    # program_template = Program_Template(p_a, rules, 300)

    dilp = DILP(language_frame, B, P, N, program_template)
    return dilp.train(steps=200)


def can_fly():
    '''
    Learn the target predicate of can_fly(X)
    which is true if X is a bird, but not an abnormal bird
    '''
    constants = ['bluebird', 'penguin', 'ostrich', 'blackbird', 'robin',
                 'sparrow', 'starling', 'chicken', 'kiwi', 'steamerduck',
                 'kakapo', 'takahe', 'weka', 'pigeon', 'swan',
                 'duck', 'goldfinch', 'woodpecker', 'bluetit', 'greattit',
                 'puddle', 'forrestcat', 'pitbull', 'goldenretriever',
                 'perser', 'bengal', 'siamese', 'sphynx', 'ragdoll', 'savannah',
                 'sibiriancat', 'greyhound', 'malteser', 'dobermann', 'rottweiler',
                 'bostonterrier', 'scottishfold', 'exotic', 'russianblue']
    B_atom = [  # birds
        Atom([Term(False, 'bluebird')], 'is_bird'),
        Atom([Term(False, 'woodpecker')], 'is_bird'),
        Atom([Term(False, 'penguin')], 'is_bird'),
        Atom([Term(False, 'bluetit')], 'is_bird'),
        Atom([Term(False, 'greattit')], 'is_bird'),
        Atom([Term(False, 'ostrich')], 'is_bird'),
        Atom([Term(False, 'blackbird')], 'is_bird'),
        Atom([Term(False, 'robin')], 'is_bird'),
        Atom([Term(False, 'sparrow')], 'is_bird'),
        Atom([Term(False, 'starling')], 'is_bird'),
        Atom([Term(False, 'chicken')], 'is_bird'),
        Atom([Term(False, 'kiwi')], 'is_bird'),
        Atom([Term(False, 'steamerduck')], 'is_bird'),
        Atom([Term(False, 'kakapo')], 'is_bird'),
        Atom([Term(False, 'takahe')], 'is_bird'),
        Atom([Term(False, 'weka')], 'is_bird'),
        Atom([Term(False, 'pigeon')], 'is_bird'),
        Atom([Term(False, 'swan')], 'is_bird'),
        Atom([Term(False, 'duck')], 'is_bird'),
        Atom([Term(False, 'goldfinch')], 'is_bird'),
        # abnormal birds
        Atom([Term(False, 'penguin')], 'abnormal_bird'),
        Atom([Term(False, 'ostrich')], 'abnormal_bird'),
        Atom([Term(False, 'chicken')], 'abnormal_bird'),
        Atom([Term(False, 'kiwi')], 'abnormal_bird'),
        Atom([Term(False, 'steamerduck')], 'abnormal_bird'),
        Atom([Term(False, 'kakapo')], 'abnormal_bird'),
        Atom([Term(False, 'takahe')], 'abnormal_bird'),
        Atom([Term(False, 'weka')], 'abnormal_bird')]
    B = [Literal(atom, False) for atom in B_atom]

    P_atom = [Atom([Term(False, 'bluebird')], 'target'),
              Atom([Term(False, 'blackbird')], 'target'),
              Atom([Term(False, 'robin')], 'target'),
              Atom([Term(False, 'sparrow')], 'target'),
              Atom([Term(False, 'starling')], 'target'),
              Atom([Term(False, 'pigeon')], 'target'),
              Atom([Term(False, 'swan')], 'target'),
              Atom([Term(False, 'duck')], 'target'),
              Atom([Term(False, 'goldfinch')], 'target'),
              Atom([Term(False, 'woodpecker')], 'target'),
              Atom([Term(False, 'bluetit')], 'target'),
              Atom([Term(False, 'greattit')], 'target')]
    P = [Literal(atom, False) for atom in P_atom]

    N_atom = [Atom([Term(False, 'penguin')], 'target'),
              Atom([Term(False, 'chicken')], 'target'),
              Atom([Term(False, 'kiwi')], 'target'),
              Atom([Term(False, 'steamerduck')], 'target'),
              Atom([Term(False, 'takahe')], 'target'),
              Atom([Term(False, 'puddle')], 'target'),
              Atom([Term(False, 'siamese')], 'target'),
              Atom([Term(False, 'forrestcat')], 'target'),
              Atom([Term(False, 'pitbull')], 'target'),
              Atom([Term(False, 'sphynx')], 'target'),
              Atom([Term(False, 'sibiriancat')], 'target'),
              Atom([Term(False, 'greyhound')], 'target'),
              Atom([Term(False, 'russianblue')], 'target')]
    N = [Literal(atom, False) for atom in N_atom]

    term_x_0 = Term(True, 'X_0')

    p_e = [Literal(Atom([term_x_0], 'is_bird'), False),
           Literal(Atom([term_x_0], 'abnormal_bird'), False)]
    p_a = []
    target = Literal(Atom([term_x_0], 'target'), False)

    # Define rules for intensional predicates
    p_a_rule = (None, None)
    target_rule = (Rule_Template_Negation(0, False, True),
                   None)
    rules = {target: target_rule}

    language_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    # program_template = Program_Template(p_a, rules, 300)

    dilp = DILP(language_frame, B, P, N, program_template)
    return dilp.train(steps=200)


def odd():
    '''
   Learn the target predicate of odd(X)
   which is true if X is an odd number
   '''
    B_atom = [Atom([Term(False, '0')], 'zero')] + \
             [Atom([Term(False, str(i)), Term(False, str(i + 1))], 'succ')
              for i in range(0, 20)]
    B = [Literal(atom, False) for atom in B_atom]
    # Odd numbers
    P_atom = [Atom([Term(False, str(i))], 'target') for i in range(1, 21, 2)]
    P = [Literal(atom, False) for atom in P_atom]
    # Even numbers
    N_atom = [Atom([Term(False, str(i))], 'target') for i in range(0, 21, 2)]
    N = [Literal(atom, False) for atom in N_atom]

    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')

    p_e = [Literal(Atom([term_x_0], 'zero'), False), Literal(Atom([term_x_0, term_x_1], 'succ'), False)]
    p_a = [Literal(Atom([term_x_0, term_x_1], 'succ2'), False),
           Literal(Atom([term_x_0], 'even'), False)]
    target = Literal(Atom([term_x_0], 'target'), False)
    constants = [str(i) for i in range(0, 21)]

    # Define rules for intensional predicates
    succ2_rule = (Rule_Template_Negation(1, False, False), None)
    even_rule = (Rule_Template_Negation(0, False, False),
                 Rule_Template_Negation(1, True, False))
    target_rule = (Rule_Template_Negation(0, True, True), None)
    rules = {p_a[0]: succ2_rule, p_a[1]: even_rule, target: target_rule}

    language_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    # program_template = Program_Template(p_a, rules, 300)

    dilp = DILP(language_frame, B, P, N, program_template)

    loss = dilp.train(steps=400)
    dilp.show_definition()
    return loss


best_loss = 99999
for i in range(100):
    print(f"Iteration {i}")
    loss = no_negative_cycle()
    if loss < best_loss:
        best_loss = loss
    print(f"Lowest loss: {best_loss}")

print(f"Lowest loss: {loss}")
