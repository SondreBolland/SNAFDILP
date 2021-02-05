'''Defines the extended rule template including a parameter for negation
'''


class Rule_Template_Negation:

    def __init__(self, v: int, allow_intensional: bool, neg: bool):
        '''
        Arguments:
            v {int} -- numberof existentially quantified variable allowed in the clause
            allow_intensional {bool} -- True is intensional predicates are allowed, False if only extensional predicates
            neg {bool} -- True if negative literals are allowed, False if only positive literals are allowed
        '''

        self._v = v
        self._allow_intensional = allow_intensional
        self._neg = neg

    @property
    def v(self):
        return self._v

    @property
    def allow_intensional(self):
        return self._allow_intensional

    @property
    def neg(self):
        return self._neg
