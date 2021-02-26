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
        Arguments:
            program {list} -- List of clauses defining all relations in the logic program
        '''
        self.program = program
        self.edge_list = []
        self.graph = self.generate_dependency_graph()

    def generate_dependency_graph(self):
        '''
        Iterate through all clauses of the program and add all predicates as nodes
        and refers to relations as edges
        :return: Graph: dependency graph of the program
        '''
        graph = nx.MultiDiGraph()
        if len(self.program) == 0:
            return graph

        # For each clause in the program fetch the edges "refers to"
        for clause in self.program:
            if type(clause) != Clause:
                continue
            edge1, edge2 = self.get_edge_from_clause(clause)
            self.add_edge(edge1, graph)
            self.add_edge(edge2, graph)

        return graph

    def add_edge(self, edge, graph):
        '''
        Add edge if not already in graph.
        Negative edges override positive edges.
        '''
        if edge.negated:
            positive_edge = Edge(edge.source, edge.target, False)
            # remove all positive instances of the edge
            while positive_edge in self.edge_list:
                self.edge_list.remove(positive_edge)
        # If the edge is not in the graph, nor its negative version, add edge
        if edge not in self.edge_list and not self.has_negative_edge_version(edge):
            self.edge_list.append(edge)
            graph.add_edge(edge.source, edge.target, negated=edge.negated,
                           color='blue' if edge.negated else 'black')

    def has_negative_edge_version(self, edge):
        positive_edge = Edge(edge.source, edge.target, True)
        return positive_edge in self.edge_list

    def get_edge_from_clause(self, clause: Clause):
        '''
        Generates edges from clause. head predicate is source
        and body predicates are targets. Each clause has two bodies hence two edges.
        Each edge has an attribute "negated" which refers to
        the negation of the body's literals.
        :param clause: Clause object to retrieve edges from
        :return: two Edge objects from the clause
        '''
        head_predicate = clause.head.predicate
        body1 = clause.body[0]
        body2 = clause.body[1]
        if not type(body1) == Literal and not type(body2) == Literal:
            raise ValueError("The components of the clause must be literals.")

        body1_predicate = body1.predicate
        body2_predicate = body2.predicate

        edge1 = Edge(head_predicate, body1_predicate, body1.negated)
        edge2 = Edge(head_predicate, body2_predicate, body2.negated)

        return edge1, edge2

    def is_stratified(self):
        '''
        Checks if the logic program can be stratified.
        :return: bool: True if the program can be stratified. False if not.
        '''
        G = self.graph
        try:
            cycle = nx.find_cycle(G)
            for edge in cycle:
                source = edge[0]
                target = edge[1]
                data = G.get_edge_data(source, target)
                negated = data[0]['negated']
                if negated:
                    return False
        except nx.NetworkXNoCycle:
            pass
        return True

    def will_terminate(self):
        pass

    def __str__(self):
        '''
        Does not show whether an edge is negative or positive
        '''
        G = self.graph
        string = ""
        for node in self.graph.nodes:
            string += f'{node} -> ('
            for i, target in enumerate(G[node]):
                data = G.get_edge_data(node, target)
                for attributes in data:
                    if data[attributes]['negated']:
                        string += "not "
                string += str(target)
                if i != len(G[node])-1:
                    string += ", "
            string += '), '
        return string

    def draw(self):
        '''
        Draws dependency graph.
        Does not support relexive edges.
        '''
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import to_agraph
        G = self.graph
        pos = nx.circular_layout(G)
        edges = G.edges()
        colors = nx.get_edge_attributes(G, 'color').values()

        nx.draw(G, pos, edges=edges, node_color='red', edge_color=colors, with_labels=True)
        plt.show()


class Edge:

    def __init__(self, source, target, negated: bool):
        '''
        Arguments:
            source {String} -- Source of the edge
            target {String} -- Target of the edge
            negated {bool} -- Whether the edge is negative (represents negation)
        '''
        self._source = source
        self._target = target
        self._negated = negated

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    @property
    def negated(self):
        return self._negated

    def __str__(self):
        return f"({self.source}, {self.target}," + "{prefix})".format(prefix="negative" if self.negated else "positive")

    def __eq__(self, other):
        if type(other) != Edge:
            return False
        return self.source == other.source\
               and self.target == other.target\
               and self.negated == other.negated
