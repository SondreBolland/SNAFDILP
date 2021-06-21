
'''
Datatype to determine the clauses which can be generated for predicate
E.g.:
name = target
disallowed_predicates = [pred1]
disallowed_negated_predicate = [pred2]
With these parameters clauses with pred1 in the body are not allowed.
Nor are clauses with pred2 negated.
target <-- pred1, pred1 (Not allowed)
target <-- not pred2, not pred2 (Not allowed)
target <-- pred2, pred2 (Allowed)

'''
class Clause_Generation_Parameter:

    def __init__(self, predicate_name, stratum,
                 disallowed_predicates: list, disallowed_negated_predicates: list):
        self._name = predicate_name
        self._stratum = stratum
        self._disallowed_predicates = disallowed_predicates
        self._disallowed_negated_predicates = disallowed_negated_predicates

    def can_reference(self, other_predicate_name, negated=False):
        '''
        Based of the parameters, determines whether the predicate can
        have a clause which references <code>other_predicate_name</code>
        :param other_predicate_name: predicate to reference
        :param negated: if <code>other_predicate_name</code> is negated or not
        :return: True if it can reference, False if not
        '''
        if other_predicate_name in self._disallowed_predicates:
            return False
        if negated:
            return other_predicate_name not in self._disallowed_negated_predicates
        return True


    @property
    def name(self):
        return self._name

    @property
    def stratum(self):
        return self._stratum

    @property
    def disallowed_predicates(self):
        return self._disallowed_predicates

    @property
    def disallowed_negated_predicates(self):
        return self._disallowed_negated_predicates

    def __str__(self):
        return f"{self.name}. Disallowed: {str(self.disallowed_predicates)} Negated: {str(self.disallowed_negated_predicates)}"

    def __lt__(self, other):
        return self.stratum < other.stratum

    def __gt__(self, other):
        return self.stratum > other.stratum
