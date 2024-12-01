def f(x):
    return 3*x**2-4*x+5

def f_d(x):
    return 6*x-4

def L(x):
    h = 0.00001
    return (f(x+h)-f(x)) / h # we devided over h to normalize the function

x = 5
print(f(x))
print(f_d(x))
print(L(x))

print('----------')

a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c

h = 0.000001

c+=h
d2 = a*b + c
print('d1', d1)
print('d2', d2)
print('slope', '+'if (d2-d1)/h>=0 else '-', abs((d2-d1)/h))

print('----------')

'''
Start building micrograd
'''

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, _children={self._prev}, _op='{self._op}', label='{self.label}')";

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

a = Value(2.0, 'a')
b = Value(-3.0, 'b')
c = Value(10.0, 'c')
print(a)
print(a+b)
print(a*b)
d = a*b+c
d.label = 'd'

print(d)
