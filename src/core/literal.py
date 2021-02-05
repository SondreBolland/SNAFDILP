'''Defines the literal class
'''
import collections

from src.core import Atom


class Literal:
    def __init__(self, atom: Atom, negated: bool):
        '''
        Arguments:
            atom {Atom} -- atom of the literal
            negated {bool} -- true if the atom is negated and false if the atom is not negated
        '''
        self._atom = atom
        self._negated = negated

    def __str__(self):
        return "{prefix} ".format(prefix="not" if self._negated else "") + str(self._atom)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self._negated == other.negated and self._atom == other.atom

    def __hash__(self):
        return hash(str(self))

    @property
    def negated(self):
        return self._negated

    @negated.setter
    def terms(self, value):
        self._negated = value

    @property
    def atom(self):
        return self._atom

    @atom.setter
    def predicate(self, value):
        self._atom = value

    def is_same_predicate(self, other):
        '''Checks equality of predicate

        Arguments:
            other {Atom} -- Atom to check predicate equality
        '''

        return self._atom.predicate == other.atom.predicate
