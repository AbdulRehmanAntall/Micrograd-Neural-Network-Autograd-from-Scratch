import math
import matplotlib.pyplot as plt
from src.engine import Value

# -------------------------------
# Example: Function plotting
# -------------------------------

def f(x):
    return 3*x**2 - 4*x + 5

xs = [x*0.25 for x in range(-20, 20)]  # -5 to 5 with step 0.25
ys = [f(x) for x in xs]

plt.plot(xs, ys)
plt.title("f(x) = 3x^2 - 4x + 5")
plt.show()

# -------------------------------
# Example: Numerical derivative
# -------------------------------

x = 2/3
h = 1e-6
slope = (f(x+h) - f(x))/h
print("Slope at x=2/3:", slope)

# -------------------------------
# Example: Value class forward/backward
# -------------------------------

a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = a*b + 10
c.backward()

print("a.grad:", a.grad)
print("b.grad:", b.grad)
print("c.data:", c.data)

# -------------------------------
# Example: graphviz computation graph
# -------------------------------

from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir':'LR'})
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape='record'
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

# Demo forward/backward graph
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b  = Value(6.88, label='b')

y = (x1*w1 + x2*w2 + b).tanh()
y.label='y'
y.backward()

dot = draw_dot(y)
dot.render("graph_output")  # will create graph_output.svg