'''Defines the main differentiable ILP code
'''
import operator

from src.core import Literal, Atom, Term
from src.ilp import ILP, Program_Template, Language_Frame, Rule_Template, Inference, ILP_Negation
from src.ilp.generate_rules import Optimized_Combinatorial_Generator, Stratified_Program
import tensorflow as tf
from collections import OrderedDict
import numpy as np

from src.ilp.generate_rules.optimized_combinatorial_negation import Optimized_Combinatorial_Generator_Negation
from src.ilp.generate_rules.optimized_combinatorial_stratified import Optimized_Combinatorial_Generator_Stratified
from src.utils import printProgressBar
from src.utils import create_mask
# import tensorflow.contrib.eager as tfe # obsolete in TF2
import os


class DILP():

    def __init__(self, language_frame: Language_Frame, background: list, positive: list, negative: list,
                 program_template: Program_Template, clause_parameters: dict={}):
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
        self.clause_parameters = clause_parameters
        self.training_data = OrderedDict()  # index to label
        self.__init__parameters()

    def __init__parameters(self):
        self.rule_weights = OrderedDict()
        ilp = ILP_Negation(self.language_frame, self.background,
                           self.positive, self.negative, self.program_template)
        (valuation, valuation_mapping, predicate_valuation_idx_map) = ilp.convert()
        self.uses_negation = ilp.uses_negation
        self.valuation_mapping = valuation_mapping
        self.predicate_valuation_idx_map = predicate_valuation_idx_map
        self.base_valuation = valuation
        self.prev_valuation = valuation
        self.n_literals = len(valuation)
        self.deduction_map = {}
        self.clause_map = {}
        # Sort p_a with respect to strata
        self.clause_parameters = dict(sorted(self.clause_parameters.items(), key=lambda item: item[1]))
        p_a = list(self.clause_parameters.keys()).copy()
        if len(p_a) != 0:
            p_a.remove(self.language_frame.target)
            self.strata = self.get_strata()
        else:
            self.strata = [[self.language_frame.target]]
        self.program_template.p_a = p_a

        n_generated_clauses = 0
        with tf.compat.v1.variable_scope("rule_weights", reuse=tf.compat.v1.AUTO_REUSE):
            for p in self.program_template.p_a + [self.language_frame.target]:
                rule_manager = Optimized_Combinatorial_Generator_Stratified(
                    self.program_template.p_a + [self.language_frame.target], self.program_template.rules[p], p,
                    self.language_frame.p_e, self.clause_parameters)
                generated = rule_manager.generate_clauses()
                print(generated)
                n_generated_clauses += len(generated[0])
                n_generated_clauses += len(generated[1])

                self.clause_map[p] = generated
                self.rule_weights[p] = tf.compat.v1.get_variable(p.predicate + "_rule_weights",
                                                                 [len(generated[0]), len(
                                                                     generated[1])],
                                                                 initializer=tf.compat.v1.random_normal_initializer,
                                                                 dtype=tf.float32)
                deduction_matrices = []
                elm1 = []
                for clause1 in generated[0]:
                    elm1.append(Inference.x_c(
                        clause1, valuation_mapping, self.language_frame.constants))
                elm2 = []
                for clause2 in generated[1]:
                    elm2.append(Inference.x_c(
                        clause2, valuation_mapping, self.language_frame.constants))
                deduction_matrices.append((elm1, elm2))
                self.deduction_map[p] = deduction_matrices
        print(f"{n_generated_clauses} generated clauses")
        self.generate_training_data(valuation_mapping, self.positive, self.negative)

    def generate_training_data(self, valuation_mapping, positive, negative):
        self.training_data = OrderedDict()
        for literal in valuation_mapping:
            if literal.negated:
                positive_version = literal.__copy__()
                positive_version.negate()
                if positive_version in positive:
                    self.training_data[valuation_mapping[literal]] = 0.0
                elif positive_version in negative:
                    self.training_data[valuation_mapping[literal]] = 1.0
            else:
                if literal in positive:
                    self.training_data[valuation_mapping[literal]] = 1.0
                elif literal in negative:
                    self.training_data[valuation_mapping[literal]] = 0.0

    def get_strata(self):
        parameters = self.clause_parameters
        final_stratum = 0
        for parameter in parameters.values():
            stratum = parameter.stratum
            if stratum > final_stratum:
                final_stratum = stratum
        strata = [[]]*final_stratum
        for literal, parameter in parameters.items():
            stratum = parameter.stratum
            strata[stratum-1].append(literal)
        return strata

    def __all_variables(self):
        return [weights for weights in self.rule_weights.values()]

    def show_atoms(self, valuation):
        result = {}
        for atom in self.valuation_mapping:
            if atom in self.positive:
                print('%s Expected: 1 %.3f' %
                      (str(atom), valuation[self.valuation_mapping[atom]]))
            elif atom in self.negative:
                print('%s Expected: 0 %.3f' %
                      (str(atom), valuation[self.valuation_mapping[atom]]))

    def show_definition(self):
        for predicate in self.rule_weights:
            shape = self.rule_weights[predicate].shape
            rule_weights = tf.reshape(self.rule_weights[predicate], [-1])
            weights = tf.reshape(tf.nn.softmax(rule_weights)[:, None], shape)
            print('----------------------------')
            print(str(predicate))
            clauses = self.clause_map[predicate]
            pos = np.unravel_index(
                np.argmax(weights, axis=None), weights.shape)
            print(clauses[0][pos[0]])
            print(clauses[1][pos[1]])

            '''
            for i in range(len(indexes[0])):
                if(weights[indexes[0][i], indexes[1][i]] > max_weights):
                    max_weights = weights[indexes[0][i],
                                          indexes[1][i]] > max_weights
                print(clauses[0][indexes[0][i]])
                print(clauses[1][indexes[1][i]])
            '''
            print('----------------------------')

    def train(self, steps=501, name='test'):
        """
        :param steps:
        :param name:
        :return: the loss history
        """
        str2weights = {str(key): value for key,
                                           value in self.rule_weights.items()}
        # if name:
        #     checkpoint = tf.train.Checkpoint(**str2weights)
        #     checkpoint_dir = "./model/" + name
        #     checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        #     try:
        #         checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        #     except Exception as e:
        #         print(e)

        losses = []
        optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.05)

        for i in range(steps):
            grads = self.grad()
            optimizer.apply_gradients(zip(grads, self.__all_variables()),
                                      global_step=tf.compat.v1.train.get_or_create_global_step())
            loss_avg = float(self.loss().numpy())
            losses.append(loss_avg)
            print("-" * 20)
            print("step " + str(i) + " loss is " + str(loss_avg))
            if i % 5 == 0:
                # self.show_definition()
                deduction = self.deduction()
                # for idx, value in enumerate(deduction):
                #    print(f'{idx}: {value}')
                self.show_atoms(deduction)
                self.show_definition()
                # if name:
                # checkpoint.save(checkpoint_prefix)
                # pd.Series(np.array(losses)).to_csv(name + ".csv")
            print("-" * 20 + "\n")
        return loss_avg

    def loss(self, batch_size=-1):
        labels = np.array(
            [val for val in self.training_data.values()], dtype=np.float32)
        keys = np.array(
            [val for val in self.training_data.keys()], dtype=np.int32)
        outputs = tf.gather(self.deduction(), keys)
        if batch_size > 0:
            index = np.random.randint(0, len(labels), batch_size)
            labels = labels[index]
            outputs = tf.gather(outputs, index)
        loss = -tf.reduce_mean(input_tensor=labels * tf.math.log(outputs + 1e-10) +
                                            (1 - labels) * tf.math.log(1 - outputs + 1e-10))
        return loss

    def grad(self):
        with tf.GradientTape() as tape:
            loss_value = self.loss(-1)
            weight_decay = 0.0
            regularization = 0
            for weights in self.__all_variables():
                weights = tf.nn.softmax(weights)
                regularization += tf.reduce_sum(input_tensor=tf.sqrt(weights)
                                                ) * weight_decay
            loss_value += regularization / len(self.__all_variables())
        return tape.gradient(loss_value, self.__all_variables())

    @staticmethod
    def update_progress(progress):
        print('\r[{0}] {1}%'.format('#' * (int(progress) / 10), progress))

    def deduction(self):
        # takes background as input and return a valuation of target ground atoms
        valuation = self.base_valuation
        print('Performing Inference')
        for stratum in self.strata:
            for step in range(self.program_template.T):
                printProgressBar(step, self.program_template.T, prefix='Progress:',
                                 suffix='Complete', length=50)
                valuation = self.inference_step(valuation, stratum)
        print('Inference Complete')
        return valuation

    def inference_step(self, valuation, stratum):
        deduced_valuation = tf.zeros(valuation.shape[0])
        for predicate in stratum:
            valuation_idx_slice = self.predicate_valuation_idx_map[predicate.predicate]
            for matrix in self.deduction_map[predicate]:
                deduced_valuation += DILP.inference_single_predicate(
                    valuation, matrix, self.rule_weights[predicate])
                deduced_valuation = self.update_negated_literals(deduced_valuation, valuation_idx_slice)

        c_t = deduced_valuation + valuation - deduced_valuation * valuation
        #con_s = c_t + deduced_valuation - c_t * deduced_valuation
        # Update negated literals for valuations not updated in the predicate inference
        all_values = [0, int(self.n_literals / 2) - 1]
        con_s = self.update_negated_literals(c_t, all_values)
        return con_s

    def update_negated_literals(self, valuation, valuation_slice):
        '''
            All valuations that have been changed have their negation
            updated.
        '''
        if not self.uses_negation:
            return valuation

        change_indices = [1] * len(valuation)
        updated_valuation = [0] * len(valuation)
        n_positive_literals = len(valuation) / 2

        start = valuation_slice[0]
        end = valuation_slice[1] + 1
        idx = start
        neg_idx = int(idx + n_positive_literals)
        for value in valuation[start:end]:
            updated_valuation[idx] = value
            updated_valuation[neg_idx] = self.negation_as_failure(value)
            change_indices[idx] = 0
            change_indices[neg_idx] = 0
            idx += 1
            neg_idx += 1

        mask = create_mask(valuation, change_indices)
        new_valuation = valuation * mask + updated_valuation * (1 - mask)
        return new_valuation

    def negation_as_failure(self, value):
        '''
        Negates the given value
        :param value: value to negate
        :return: negated value in interval [0,1]
        '''
        # Strong negation
        return 1.0 - value
        # Weak negation
        #return 1.0 if value == 0.0 else 0.0
        # Weak negation with threshold
        #threshold = 0.6
        #return 1.0 if value < threshold else 0.0

    def print_v(self, valuation):
        n_positive_literals = int(self.n_literals / 2)
        for idx, value in enumerate(valuation):
            if idx == n_positive_literals:
                print('-----Negatives-----')
            print(f'{idx}: {value}')

    @staticmethod
    def inference_single_predicate(valuation, deduction_matrices, rule_weights):
        '''
        :param valuation:
        :param deduction_matrices: list of list of matrices
        :param rule_weights: list of tensor, shape (number_of_rule_temps, number_of_clauses_generated)
        :return:
        '''
        result_valuations = [[], []]
        for i in range(len(result_valuations)):
            for matrix in deduction_matrices[i]:
                result_valuations[i].append(
                    DILP.inference_single_clause(valuation, matrix))

        c_p = []  # flattened
        for clause1 in result_valuations[0]:
            for clause2 in result_valuations[1]:
                c_p.append(tf.maximum(clause1, clause2))

        rule_weights = tf.reshape(rule_weights, [-1])
        prob_rule_weights = tf.nn.softmax(rule_weights)[:, None]
        return tf.reduce_sum(input_tensor=(tf.stack(c_p) * prob_rule_weights), axis=0)

    @staticmethod
    def inference_single_clause(valuation, X):
        '''
        The F_c in the paper
        :param valuation:
        :param X: array, size (number)
        :return: tensor, size (number_of_ground_atoms)
        '''
        X1 = X[:, :, 0, None]
        X2 = X[:, :, 1, None]
        Y1 = tf.gather_nd(params=valuation, indices=X1)
        Y2 = tf.gather_nd(params=valuation, indices=X2)
        Z = Y1 * Y2
        return tf.reduce_max(input_tensor=Z, axis=1)
