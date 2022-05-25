class Node:
    def __init__(self, value, parent=None, children=None):
        self.value = value
        self.parent = parent
        self.children = [] if children is None else children
