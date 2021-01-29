from src.ilp import Rule_Template
'''Defines the extended rule template including a parameter for negation
'''


class Rule_Template_Negation(Rule_Template):

    def __init__(self, neg: bool):
        '''
        Arguments:
            neg {bool} -- True if negative literals are allowed, False if only positive literals are allowed
        '''

        self._neg = neg

    @property
    def neg(self):
        return self._neg
