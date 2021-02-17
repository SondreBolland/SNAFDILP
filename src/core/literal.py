'''Defines the literal class
'''
import collections

from src.core import Atom


class Literal(Atom):

    def __init__(self, *args):
        '''
        Arguments:
            terms {list} -- List of of terms defining atom
            name {str} -- name of the predicate
            negated {bool} -- true if the atom is negated and false if the atom is not negated
        '''
        if len(args) == 3:
            terms = args[0]
            predicate = args[1]
            negated = args[2]
        elif len(args) == 2:
            atom = args[0]
            terms = atom.terms
            predicate = atom.predicate
            negated = args[1]
        else:
            raise ValueError("Literal requires (terms, predicate, negated) or (atom, negated)")
        Atom.__init__(self, terms, predicate)
        self._negated = negated

    def __str__(self):
        return "{prefix} ".format(prefix="not" if self._negated else "") +  super(Literal, self).__str__()

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        '''
        :param other: Atom or Literal
        :return: true if the atoms of the literals are equal
        and they're both negated (or both not)
        '''
        equals = True
        if isinstance(other, Literal):
            equals = equals and self._negated == other.negated
        else:
            equals = equals and not self._negated
        equals = equals and super(Literal, self).__eq__(other)
        return equals

    def __hash__(self):
        return hash(str(self))

    @property
    def negated(self):
        return self._negated
