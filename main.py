'''For testing DILP
'''
from src.core import Term, Atom, Literal
from src.ilp import Language_Frame, Program_Template, Rule_Template
from src.dilp import DILP
import tensorflow as tf

from src.ilp.template.rule_template_negation import Rule_Template_Negation

tf.compat.v1.enable_eager_execution()


def even_numbers_test():
    B = [Atom([Term(False, '0')], 'zero')] + \
        [Atom([Term(False, str(i)), Term(False, str(i + 1))], 'succ')
         for i in range(0, 20)]

    P = [Atom([Term(False, str(i))], 'target') for i in range(0, 21, 2)]
    N = [Atom([Term(False, str(i))], 'target') for i in range(1, 21, 2)]

    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')

    p_e = [Atom([term_x_0], 'zero'), Atom([term_x_0, term_x_1], 'succ')]
    p_a = [Atom([term_x_0, term_x_1], 'pred')]
    target = Atom([term_x_0], 'target')
    constants = [str(i) for i in range(0, 21)]

    # Define rules for intensional predicates
    p_a_rule = (Rule_Template(1, False), None)
    target_rule = (Rule_Template(0, False), Rule_Template(1, True))
    rules = {p_a[0]: p_a_rule, target: target_rule}

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    #program_template = Program_Template(p_a, rules, 300)

    dilp = DILP(langage_frame, B, P, N, program_template)

    dilp.train()


def even_numbers_negation_test():
    B_atom = [Atom([Term(False, '0')], 'zero')] + \
        [Atom([Term(False, str(i)), Term(False, str(i + 1))], 'succ')
         for i in range(0, 20)]
    B = [Literal(atom, False) for atom in B_atom]

    P_atom = [Atom([Term(False, str(i))], 'target') for i in range(0, 21, 2)]
    positive_literals_P = [Literal(atom, False) for atom in P_atom]
    N_atom = [Atom([Term(False, str(i))], 'target') for i in range(1, 21, 2)]
    positive_literals_N = [Literal(atom, False) for atom in N_atom]

    # If target(0) is in the list of positive examples, then
    # not target(0) must be in the list of negative examples
    negative_literals_P = [Literal(atom, True) for atom in N_atom]
    negative_literals_N = [Literal(atom, True) for atom in P_atom]

    P = positive_literals_P + negative_literals_P
    N = positive_literals_N + negative_literals_N

    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')

    p_e = [Literal(Atom([term_x_0], 'zero'), False), Literal(Atom([term_x_0, term_x_1], 'succ'), False)]
    p_a = [Literal(Atom([term_x_0, term_x_1], 'pred'), False)]
    target = Literal(Atom([term_x_0], 'target'), False)
    constants = [str(i) for i in range(0, 21)]

    # Define rules for intensional predicates
    p_a_rule = (Rule_Template_Negation(1, False, False), None)
    target_rule = (Rule_Template_Negation(0, False, True),
                   Rule_Template_Negation(1, True, False))
    rules = {p_a[0]: p_a_rule, target: target_rule}

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    #program_template = Program_Template(p_a, rules, 300)

    dilp = DILP(langage_frame, B, P, N, program_template)

    loss = dilp.train(steps=300)
    dilp.show_definition()
    return loss


def orphan():
    '''
    Learn the target predicate of orphan(X)
    which is true if X has no parent
    '''
    constants = ['Paul', 'Randy', 'Rachel', 'Bob', 'Alice',
                 'Stan', 'Kyle', 'Peter', 'Tony', 'Rachel', 'Monica']
    B_atom = [Atom([Term(False, 'Bob'), Term(False, 'Paul')], 'parent'),
              Atom([Term(False, 'Bob'), Term(False, 'Alice')], 'parent'),
              Atom([Term(False, 'Randy'), Term(False, 'Tony')], 'parent'),
              Atom([Term(False, 'Randy'), Term(False, 'Stan')], 'parent'),
              Atom([Term(False, 'Peter'), Term(False, 'Kyle')], 'parent'),
              Atom([Term(False, 'Alice'), Term(False, 'Kyle')], 'parent')]
    B = [Literal(atom, False) for atom in B_atom]

    P_atom = [Atom([Term(False, 'Randy')], 'target'),
              Atom([Term(False, 'Bob')], 'target'),
              Atom([Term(False, 'Peter')], 'target'),
              Atom([Term(False, 'Rachel')], 'target'),
              Atom([Term(False, 'Monica')], 'target')]
    positive_literal_P = [Literal(atom, False) for atom in P_atom]

    N_atom = [Atom([Term(False, 'Paul')], 'target'),
              Atom([Term(False, 'Alice')], 'target'),
              Atom([Term(False, 'Kyle')], 'target'),
              Atom([Term(False, 'Tony')], 'target'),
              Atom([Term(False, 'Stan')], 'target')]
    positive_literal_N = [Literal(atom, False) for atom in N_atom]

    negative_literal_P = [Literal(atom, True) for atom in N_atom]
    negative_literal_N = [Literal(atom, True) for atom in P_atom]

    P = positive_literal_P + negative_literal_P
    N = positive_literal_N + negative_literal_N

    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')

    p_e = [Atom([term_x_0, term_x_1], 'parent')]
    p_a = []
    target = Atom([term_x_0], 'target')

    # Define rules for intensional predicates
    p_a_rule = (None, None)
    target_rule = (Rule_Template_Negation(1, False, True),
                   None)
    rules = {target: target_rule}

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    #program_template = Program_Template(p_a, rules, 300)

    dilp = DILP(langage_frame, B, P, N, program_template)
    dilp.train(steps=200)

def no_ancestor():
    '''
    Learn the target predicate of ancestor(X)
    which is true if X has no ancestor
    '''
    constants = ['Paul', 'Randy', 'Rachel', 'Bob', 'Alice',
                 'Stan', 'Kyle', 'Peter', 'Tony', 'Monica', 'Jessica', 'Duncan']
    B_atom = [Atom([Term(False, 'Bob'), Term(False, 'Paul')], 'parent'),
              Atom([Term(False, 'Bob'), Term(False, 'Alice')], 'parent'),
              Atom([Term(False, 'Randy'), Term(False, 'Tony')], 'parent'),
              Atom([Term(False, 'Randy'), Term(False, 'Stan')], 'parent'),
              Atom([Term(False, 'Peter'), Term(False, 'Kyle')], 'parent'),
              Atom([Term(False, 'Alice'), Term(False, 'Kyle')], 'parent'),
              Atom([Term(False, 'Jessica'), Term(False, 'Bob')], 'parent'),
              Atom([Term(False, 'Jessica'), Term(False, 'Randy')], 'parent'),
              Atom([Term(False, 'Matthew'), Term(False, 'Jessica')], 'parent')]

    B = [Literal(atom, False) for atom in B_atom]

    P_atom = [Atom([Term(False, 'Peter')], 'target'),
              Atom([Term(False, 'Monica')], 'target'),
              Atom([Term(False, 'Rachel')], 'target'),
              Atom([Term(False, 'Jessica')], 'target'),
              Atom([Term(False, 'Duncan')], 'target'),
              Atom([Term(False, 'Matthew')], 'target')]
    positive_literal_P = [Literal(atom, False) for atom in P_atom]

    N_atom = [Atom([Term(False, 'Randy')], 'target'),
              Atom([Term(False, 'Bob')], 'target'),
              Atom([Term(False, 'Alice')], 'target'),
              Atom([Term(False, 'Kyle')], 'target'),
              Atom([Term(False, 'Tony')], 'target'),
              Atom([Term(False, 'Stan')], 'target'),
              Atom([Term(False, 'Paul')], 'target')]
    positive_literal_N = [Literal(atom, False) for atom in N_atom]

    negative_literal_P = [Literal(atom, True) for atom in N_atom]
    negative_literal_N = [Literal(atom, True) for atom in P_atom]

    P = positive_literal_P + negative_literal_P
    N = positive_literal_N + negative_literal_N

    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')

    p_e = [Atom([term_x_0, term_x_1], 'parent')]
    p_a = []
    target = Atom([term_x_0], 'target')

    # Define rules for intensional predicates
    p_a_rule = (None, None)
    target_rule = (Rule_Template_Negation(1, False, True),
                   None)
    rules = {target: target_rule}

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    #program_template = Program_Template(p_a, rules, 300)

    dilp = DILP(langage_frame, B, P, N, program_template)
    dilp.train(steps=200)

def prdecessor():
    B = [Atom([Term(False, '0')], 'zero')] + \
        [Atom([Term(False, str(i)), Term(False, str(i + 1))], 'succ')
         for i in range(0, 9)]

    P = [Atom([Term(False, str(i + 1)), Term(False, str(i))], 'target')
         for i in range(0, 9)]
    N = []
    for i in range(0, 10):
        for j in range(0, 10):
            if j != i + 1:
                N.append(
                    Atom([Term(False, str(j)), Term(False, str(i))], 'target'))
    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')

    p_e = [Atom([term_x_0], 'zero'), Atom([term_x_0, term_x_1], 'succ')]
    p_a = []

    target = Atom([term_x_0, term_x_1], 'target')
    # target_rule = (Rule_Template(0, False), Rule_Template(1, True))
    target_rule = (Rule_Template(0, False), None)
    rules = {target: target_rule}
    constants = [str(i) for i in range(0, 10)]

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    dilp = DILP(langage_frame, B, P, N, program_template)
    dilp.train()


def less_than():
    B = [Atom([Term(False, '0')], 'zero')] + \
        [Atom([Term(False, str(i)), Term(False, str(i + 1))], 'succ')
         for i in range(0, 9)]

    P = []
    N = []
    for i in range(0, 10):
        for j in range(0, 10):
            if j >= i:
                N.append(
                    Atom([Term(False, str(j)), Term(False, str(i))], 'target'))
            else:
                P.append(
                    Atom([Term(False, str(j)), Term(False, str(i))], 'target'))

    term_x_0 = Term(True, 'X_0')
    term_x_1 = Term(True, 'X_1')

    p_e = [Atom([term_x_0], 'zero'), Atom([term_x_0, term_x_1], 'succ')]
    p_a = []

    target = Atom([term_x_0, term_x_1], 'target')
    # target_rule = (Rule_Template(0, False), Rule_Template(1, True))
    target_rule = (Rule_Template(0, False), Rule_Template(1, True))
    rules = {target: target_rule}
    constants = [str(i) for i in range(0, 10)]

    langage_frame = Language_Frame(target, p_e, constants)
    program_template = Program_Template(p_a, rules, 10)
    dilp = DILP(langage_frame, B, P, N, program_template)
    dilp.train()


#even_numbers_test()
#even_numbers_negation_test()

best_loss = 99999
for i in range(100):
    print(f"Iteration {i}")
    loss = even_numbers_negation_test()
    if loss < best_loss:
        print(f"Lowest loss: {loss}")
        best_loss = loss

print(f"Lowest loss: {loss}")

#less_than()
#prdecessor()
#orphan()
#no_ancestor()

