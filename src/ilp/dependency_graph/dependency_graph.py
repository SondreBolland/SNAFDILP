from src.core import Clause, Literal
import networkx as nx

'''
Graph representation of logic program where each predicate is a node
and the relation *refers to* is represented by edges. 
A relation p refers to relation q if there exists a clause with head p and body q.
'''
class Dependency_Graph:

    def __init__(self, program: list):
        '''
        :param program: list of clauses defining all relations in the logic program
        '''
        if len(program) != 0:
            if type(program[0]) != Clause:
                raise ValueError("Needs to be a list of clauses")
        self.program = program
        self.graph = self.generate_dependency_graph()

    def generate_dependency_graph(self):
        '''
        Iterate through all clauses of the program and add all predicates as nodes
        and refers to relations as edges
        :return: Graph: dependency graph of the program
        '''
        graph = nx.Graph()
        if len(self.program) == 0:
            return graph

        for clause in self.program:
            edge1, edge2 = self.get_edge_from_clause(clause)
            graph.add_edge(edge1)
            graph.add_edge(edge2)

    def get_edge_from_clause(self, clause: Clause):
        '''
        Generates edges from clause. head predicate is source
        and body predicates are targets.
        Each edge has an attribute "negated" which refers to
        the negation of the body's literals.
        :param clause:
        :return:
        '''
        head_predicate = clause.head.predicate
        body1 = clause.body[0]
        body2 = clause.body[1]
        if not type(body1) == Literal and not type(body2) == Literal:
            raise ValueError("The components of the clause must be literals.")

        body1_predicate = body1.predicate
        body2_predicate = body2.predicate

        edge1 = (head_predicate, body1_predicate, {'negated': body1.negated})
        edge2 = (head_predicate, body2_predicate, {'negated': body2.negated})
        return edge1, edge2

    def is_statified(self):
        '''
        Checks if the logic program can be stratified.
        :return: bool: True if the program can be stratified. False if not.
        '''
        pass

    def will_terminate(self):
        pass

    def __str__(self):
        string = ""
        for node in self.graph.nodes:
            string += '%s -> %s' % (str(node), ','.join(str(target) for target in self.graph[node]))
        return string
