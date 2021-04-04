import DecisionLeaf
import re


def is_numeric(example, value):
    return isinstance(example[value], int) or isinstance(example[value], float)


class DecisionNode:

    def __init__(self, attr, attr_name=None, children=None):
        self.attr = attr
        self.attr_name = attr_name or attr
        self.children = children or {}

    def __call__(self, titles, example):
        """Classify an example traversing the tree"""
        index = titles.index(self.attr_name)
        value = example[index]
        if is_numeric(example, index): #NUMERIC
            temp = re.findall("\d+\.\d+", list(self.children.keys())[0])  #Extract the threshold value from the string
            th = float(temp[0])
            if value <= th:
                if isinstance(self.children[f'<= {th}'], DecisionLeaf.DecisionLeaf):
                    return self.children[f'<= {th}'].classify()
                else:
                    subtree = self.children[f'<= {th}']
                    return subtree(titles, example)
            else:
                if isinstance(self.children[f'> {th}'], DecisionLeaf.DecisionLeaf):
                    return self.children[f'> {th}'].classify()
                else:
                    subtree = self.children[f'> {th}']
                    return subtree(titles, example)

         #CATEGORICAL
        elif value in self.children:
            if isinstance(self.children[value], DecisionLeaf.DecisionLeaf):
                return self.children[value].classify()
            else:
                subtree = self.children[value]
                return subtree(titles, example)


    def add(self, val, subtree):
        self.children[val] = subtree


    def display(self, attr_name, value, threshold):
        if threshold is not None:
            print(f"--> Subtree for value {attr_name} {value}")
        else:
            print(f"--> Subtree for value {attr_name} = {value}")
