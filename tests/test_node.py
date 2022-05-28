import pytest

from connectx.node import Node


def test_init_tree_v1():
    root = Node("root")
    a = Node("a")
    b = Node("b")
    root.children = [a, b]
    assert root.children.index(a) != -1
    assert root.children.index(b) != -1
    assert a.parent == root
    assert b.parent == root


def test_init_tree_v2():
    a = Node("a")
    b = Node("b")
    root = Node("root", children=[a, b])
    assert root.children.index(a) != -1
    assert root.children.index(b) != -1
    assert a.parent == root
    assert b.parent == root


def test_init_tree_v3():
    a = Node("a")
    root = Node("root", children=[a])
    b = Node("b")
    root.attach_child(b)
    assert root.children.index(a) != -1
    assert root.children.index(b) != -1
    assert a.parent == root
    assert b.parent == root


def test_init_tree_v4():
    root = Node("root")
    a = Node("a")
    with pytest.raises(AttributeError):
        a.parent = root


def test_init_tree_v5():
    b = Node("b")
    a = Node("a", children=[b])
    root = Node("root", children=[a])
    assert b.parent == a
    assert a.parent == root
