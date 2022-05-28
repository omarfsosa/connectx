from typing import Any, List


class AttachedNodeError(Exception):
    pass


def _check_empty_parent(*nodes):
    for node in nodes:
        if node.parent is not None:
            raise AttachedNodeError


class Node:
    def __init__(self, value: Any, children: List["Node"] | None = None):
        self.value = value
        self._parent = None
        self._children = []

        # uses children.setter
        if children is not None:
            self.children = children

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    @children.setter
    def children(self, nodes):
        _check_empty_parent(*nodes)
        for node in nodes:
            node._parent = self

        self._children = nodes

    def attach_child(self, node):
        _check_empty_parent(node)
        node._parent = self
        self._children.append(node)

    # -- helpers
    @property
    def depth(self):
        if self.parent is None:
            return 0

        return self.parent.depth + 1

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({self.value!r})"
