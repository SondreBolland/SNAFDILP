from src.dilp import DILP
from src.ilp import Program_Template, Language_Frame
from src.ilp.generate_rules.clause_generation_parameter import Clause_Generation_Parameter


class SNAFDILP:

    def __init__(self, language_frame: Language_Frame, background: list, positive: list, negative: list,
                 program_template: Program_Template):
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
        self.dilps = self.generate_stratified_dilps()

    def generate_stratified_dilps(self):
        '''
        Construct n DILP objects. All objects have clauses seperated into
        strata to ensure stratification.
        :return:
        '''
        rules = self.program_template.rules
        n_auxiliary_predicates = len(rules)-1 # Subtract 1 for target predicate
        # If only target predicate then no risk for stratification
        if n_auxiliary_predicates <= 0:
            return [DILP(self.language_frame, self.background,
                                   self.positive, self.negative, self.program_template)]
        target = self.language_frame.target
        rules = rules.copy()
        rules.pop(target)
        keys = list(rules.keys())
        if n_auxiliary_predicates == 1:
            strata1 = {keys[0]: 1, target: 2}
            dilp1 = self.create_DILP(strata1)
            # Removed all predicates in same strata. No need
            return [dilp1]

        if n_auxiliary_predicates == 2:
            strata1 = {keys[0]: 1, keys[1]: 2, target: 3}
            #strata2 = {keys[0]: 2, keys[1]: 1, target: 3}
            #strata3 = {keys[0]: 1, keys[1]: 1, target: 2}
            #strata4 = {keys[0]: 2, keys[1]: 1, target: 2}
            #strata5 = {keys[0]: 1, keys[1]: 2, target: 2}

            dilp1 = self.create_DILP(strata1)
            #dilp2 = self.create_DILP(strata2)
            #dilp3 = self.create_DILP(strata3)
            #dilp4 = self.create_DILP(strata4)
            #dilp5 = self.create_DILP(strata5)
            return [dilp1]#, dilp2, dilp3, dilp4, dilp5]

    def create_DILP(self, strata):
        clause_parameters = self.create_clause_parameters(strata)
        return DILP(self.language_frame, self.background,
                    self.positive, self.negative, self.program_template, clause_parameters)

    def create_clause_parameters(self, strata):
        rules = self.program_template.rules

        clause_parameters = {}
        for literal, rule_templates in rules.items():
            clause_parameter = self.clause_parameter(literal, rule_templates, rules, strata)
            clause_parameters.update({literal: clause_parameter})
        return clause_parameters

    @staticmethod
    def clause_parameter(literal, rule_templates, rules, strata):
        if not SNAFDILP.defined_intentionally(rule_templates):
            return Clause_Generation_Parameter(literal.predicate, 1, [], [])

        disallowed_predicates = []
        disallowed_negated_predicates = []
        literal_stratum = strata[literal]
        for other_literal, other_rule_templates in rules.items():
            other_predicate_name = other_literal.predicate
            if literal == other_literal:
                continue
            if not SNAFDILP.defined_intentionally(other_rule_templates):
                continue
            other_literal_stratum = strata[other_literal]
            if literal_stratum == other_literal_stratum:
                disallowed_negated_predicates.append(other_predicate_name)
            if literal_stratum < other_literal_stratum:
                disallowed_predicates.append(other_predicate_name)
        return Clause_Generation_Parameter(literal.predicate, literal_stratum, disallowed_predicates, disallowed_negated_predicates)

    @staticmethod
    def defined_intentionally(rule_templates: tuple):
        for rule_template in rule_templates:
            if rule_template is None:
                continue
            if rule_template.allow_intensional:
                return True
        return False

    def train(self, steps=200, threshold=0.01):
        '''
        For each DILP object train for <code>steps</code> steps
        :param steps: number of forward passes in the system
        :param threshold: loss threshold. If under then the
        program is considered learned
        :return: lowest loss of all DILP objects trained
        '''
        self.best_dilp = None
        best_loss = 9999
        for dilp in self.dilps:
            loss = dilp.train(steps=steps)
            if loss < best_loss:
                best_loss = loss
                self.best_dilp = dilp
            if loss < threshold:
                break
        print("\n\nBest Program --------------------------------")
        deduction = self.best_dilp.deduction()
        self.best_dilp.show_atoms(deduction)
        self.best_dilp.show_definition()
        return best_loss
