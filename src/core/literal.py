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
        or, Arguments:
            atom {Atom} -- atom of the literal
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
        return "{prefix}".format(prefix="not " if self._negated else "") + super(Literal, self).__str__()

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        '''
        :param other: Atom or Literal
        :return: true if the atoms of the literals are equal
        and they're both negated (or both not)
        '''
        if type(other) == Literal:
            if self._negated != other.negated:
                return False
        elif self._negated:
            return False
        if not super(Literal, self).__eq__(other):
            return False
        return True

    def __hash__(self):
        return hash(str(self))

    @property
    def negated(self):
        return self._negated

    def negate(self):
        self._negated = not self._negated

    def __copy__(self):
        copy_atom = Atom(self._terms, self._predicate)
        copy_literal = Literal(copy_atom, self._negated)
        return copy_literal
