#!/usr/bin/env python3

from math import sqrt, sin, cos
import math
import cmath



def debug(statement):
	print("\t", statement, ':', eval(statement))
def test(assertion):
	print('[assert]', assertion)
	assert eval(assertion)
def flatten(l):
    return [ item for sublist in l for item in sublist ]
def flattent(l):
    return tuple( item for sublist in l for item in sublist )



_c = lambda a,b: (complex(a,0),complex(b,0),)
_m = lambda a,b,c,d: [[complex(a,0),complex(b,0)], [complex(c,0),complex(d,0)]]
_mul = lambda c,m: (c[0]*m[0][0] + c[1]*m[1][0], c[0]*m[0][1] + c[1]*m[1][1],)
_cmp = lambda z1,z2: cmath.isclose(z1[0],z2[0]) and cmath.isclose(z1[1],z2[1])
# compare n-vector to n-vector
_cmp2 = lambda z1,z2: all( cmath.isclose(z1[i],z2[i]) for i in range(len(z1)) )

# multiply n-vector by nxn-matrix
_mul2 = lambda c,m: tuple( sum( c[j]*m[j][i] for j in range(len(m)) ) for i in range(len(m[0])) )
# compounds an nxn matrix
def _m2(*args):
	size = int(sqrt(len(args)))
	return tuple( tuple( complex(v, 0) for v in args[x*size:x*size+size] ) for x in range(size) )

sqrt1_2 = 1/sqrt(2)

zero = a = _c(1,0)
one = b = _c(0,1)
positive = p = _c(sqrt1_2, sqrt1_2)
negative = n = _c(sqrt1_2, -sqrt1_2)



identity_matrix = _m2(
	1,0,
	0,1)
print('multiply by identity matrix:')
debug('a')
debug('b')
debug('[a[0]*complex(1,0) + a[1]*complex(0,0), a[0]*complex(0,0) + a[1]*complex(1,0)]')
debug('identity_matrix')
debug('_mul(a,identity_matrix)')
debug('_mul(b,identity_matrix)')

test('_mul(a,identity_matrix) == a')
test('_mul(b,identity_matrix) == b')
test('_mul(p,identity_matrix) == p')

NOTGATE_matrix = _m2(
	0,1,
	1,0)
def NOTGATE(z):
	return _mul2(z, NOTGATE_matrix)

print('multiply by NOT gate:')
debug('NOTGATE_matrix')
debug('_mul(a,NOTGATE_matrix)')
debug('_mul(b,NOTGATE_matrix)')


# def test(assertion):
# 	print('[assert]', assertion)
# 	assert eval(assertion)

test('NOTGATE(_c(1,0)) == _c(0,1)')
test('NOTGATE(_c(0,1)) == _c(1,0)')
test('NOTGATE(_c(-1,0)) == _c(0,-1)')
test('NOTGATE(_c(0,-1)) == _c(-1,0)')
test('NOTGATE(_c(sqrt1_2,sqrt1_2)) == _c(sqrt1_2,sqrt1_2)')

HADAMARD_matrix = _m2(
	sqrt1_2,sqrt1_2,
	sqrt1_2,-sqrt1_2)
def HADAMARDGATE(z):
	return _mul2(z, HADAMARD_matrix)

print('multiply by Hadamard gate:')
debug('HADAMARD_matrix')
debug('_mul(a,HADAMARD_matrix)')
debug('_mul(b,HADAMARD_matrix)')
debug('HADAMARDGATE(HADAMARDGATE(one))')
debug('cmath.isclose(one[1], HADAMARDGATE(HADAMARDGATE(one))[1])')


test('HADAMARDGATE(_c(1,0)) == _c(sqrt1_2,sqrt1_2)')
test('HADAMARDGATE(_c(0,1)) == _c(sqrt1_2,-sqrt1_2)')
test('HADAMARDGATE(_c(1,0)) == _c(sqrt1_2,sqrt1_2)')
test('_cmp(one, HADAMARDGATE(HADAMARDGATE(one)))')
test('_cmp(zero, HADAMARDGATE(positive))')
test('_cmp(one, HADAMARDGATE(negative))')
test('_cmp(positive, HADAMARDGATE(zero))')
test('_cmp(negative, HADAMARDGATE(one))')

print('tensor product of vectors:')
debug('[zero[0]*zero[0], zero[0]*zero[1], zero[1]*zero[0], zero[1]*zero[1]]')
debug('[a*b for b in zero for a in zero]')
debug('[a*b for b in zero for a in positive]')
debug('[a*b for b in positive for a in zero]')

# don't use
def tensor(z1,z2):
	return tuple(a*b for b in z1 for a in z2)
debug('tensor(positive,negative)')

CNOT_matrix = _m2(
	1,0,0,0,
	0,1,0,0,
	0,0,0,1,
	0,0,1,0)
flip_CNOT_matrix = _m2(
	1,0,0,0,
	0,0,0,1,
	0,0,1,0,
	0,1,0,0)
def CNOTGATE(z1, z2):
	return _mul2(tensor(z1,z2), CNOT_matrix)
print('CNOT:')
debug('CNOT_matrix')
debug('_mul2(tensor(zero,zero), CNOT_matrix)')
debug('_mul2(tensor(one,zero), CNOT_matrix)')
debug('_mul2(_mul2(tensor(one,one), CNOT_matrix), CNOT_matrix)')
debug('_mul2(tensor(positive,positive), CNOT_matrix)')
debug('_mul2(tensor(positive,zero), CNOT_matrix)')
test('CNOTGATE(zero,zero) == tensor(zero,zero)')
test('CNOTGATE(one,zero) == tensor(one,one)')
test('CNOTGATE(one,one) == tensor(one,zero)')
test('CNOTGATE(one,one) == tensor(one,zero)')
test('CNOTGATE(zero,one) == tensor(zero,one)')
test('_mul2(tensor(positive,zero), CNOT_matrix) == (complex(sqrt1_2,0), 0j, 0j, complex(sqrt1_2,0),)')

SWAP_matrix = _m2(
	1,0,0,0,
	0,0,1,0,
	0,1,0,0,
	0,0,0,1)
def SWAPGATE(z1, z2):
	return _mul2(tensor(z1,z2), SWAP_matrix)

print('SWAP:')
debug('SWAP_matrix')
test('SWAPGATE(one,one) == tensor(one,one)')
test('SWAPGATE(one,zero) == tensor(zero,one)')
test('SWAPGATE(zero,one) == tensor(one,zero)')
test('SWAPGATE(zero,zero) == tensor(zero,zero)')
test('SWAPGATE(positive,zero) == tensor(zero,positive)')
test('SWAPGATE(positive,negative) == tensor(negative,positive)')
test('SWAPGATE(one,negative) == tensor(negative,one)')

CCNOT_matrix = _m2(
	1,0,0,0,0,0,0,0,
	0,1,0,0,0,0,0,0,
	0,0,1,0,0,0,0,0,
	0,0,0,1,0,0,0,0,
	0,0,0,0,1,0,0,0,
	0,0,0,0,0,1,0,0,
	0,0,0,0,0,0,0,1,
	0,0,0,0,0,0,1,0)
def CCNOTGATE(z1, z2, z3):
	return _mul2(tensor2(z1,z2,z3), CCNOT_matrix)



# tensors n-vector by p-vector together
def tensor2(z1,*zs):
	r = z1
	for z in zs:
		r = tuple( a*b for b in r for a in z )
	return r
print('CCNOT:')
debug('CCNOT_matrix')
debug('CCNOTGATE(zero, zero, zero)')
debug('tensor2(zero, one, zero)')
test('tensor2(zero, zero, zero) == ((1+0j), 0j, 0j, 0j, 0j, 0j, 0j, 0j)')
test('tensor2(one, one, one) == (0j, 0j, 0j, 0j, 0j, 0j, 0j, (1+0j))')
test('CCNOTGATE(zero, zero, zero) == tensor2(zero,zero,zero)')
test('CCNOTGATE(one,zero,zero) == tensor2(one,zero,zero)')
test('CCNOTGATE(one,zero,zero) == tensor2(one,zero,zero)')
test('CCNOTGATE(one,one,zero) == tensor2(one,one,one)')
test('CCNOTGATE(one,one,one) == tensor2(one,one,zero)')
test('CCNOTGATE(positive,one,zero) == (0j, 0j, (0.7071067811865475+0j), 0j, 0j, 0j, 0j, (0.7071067811865475+0j))')
debug('tensor2(positive,one,zero)')
debug('tensor2(positive,one,one)')
debug('CCNOTGATE(positive,one,one)')

CSWAP_matrix = _m2(
	1,0,0,0,0,0,0,0,
	0,1,0,0,0,0,0,0,
	0,0,1,0,0,0,0,0,
	0,0,0,1,0,0,0,0,
	0,0,0,0,1,0,0,0,
	0,0,0,0,0,0,1,0,
	0,0,0,0,0,1,0,0,
	0,0,0,0,0,0,0,1)
def CSWAPGATE(z1, z2, z3):
	return _mul2(tensor2(z1,z2,z3), CSWAP_matrix)

print('CSWAP:')
debug('CSWAPGATE(one,one,zero)')
test('CSWAPGATE(one,one,zero) == tensor2(one,zero,one)')
test('CSWAPGATE(zero,one,zero) == tensor2(zero,one,zero)')
test('CSWAPGATE(one,zero,positive) == tensor2(one,positive,zero)')
test('CSWAPGATE(one,negative,positive) == tensor2(one,positive,negative)')



MATRIX1 = _m2(
	1, 0,
	0, 1)
MATRIX2 = _m2(
	4, 5,
	6, 7)

debug('MATRIX1')
debug('MATRIX2')
debug('[ [ MATRIX1[0][0] ] ]')
# flat multiply by factor, meant for matrices tensoring
_mul3 = lambda c,m: tuple( c*m[j] for j in range(len(m)) )
debug('_mul3(MATRIX1[0][0], MATRIX2[0])')
debug('_mul3(MATRIX1[0][1], MATRIX2[0])')
debug('_mul3(MATRIX1[0][0], MATRIX2[1])')
debug('_mul3(MATRIX1[0][1], MATRIX2[1])')
debug('_mul3(MATRIX1[1][0], MATRIX2[0])')
debug('_mul3(MATRIX1[1][1], MATRIX2[0])')
debug('_mul3(MATRIX1[1][0], MATRIX2[1])')
debug('_mul3(MATRIX1[1][1], MATRIX2[1])')

print('more testing:')
debug('flatten([ _mul3(MATRIX1[0][i], MATRIX2[0]) for i in range(len(MATRIX1[0])) ])')
debug('flatten([ _mul3(MATRIX1[0][i], MATRIX2[1]) for i in range(len(MATRIX1[0])) ])')
debug('flatten([ _mul3(MATRIX1[1][i], MATRIX2[0]) for i in range(len(MATRIX1[0])) ])')
debug('flatten([ _mul3(MATRIX1[1][i], MATRIX2[1]) for i in range(len(MATRIX1[0])) ])')
debug('[ flatten([ _mul3(MATRIX1[j][i], MATRIX2[k]) for i in range(len(MATRIX1[0])) ]) for j in range(len(MATRIX1)) for k in range(len(MATRIX2)) ]')

# tensors nxn*pxp matrices together
tensor_matrix = lambda m1,m2: [ flatten([ _mul3(m1[j][i], m2[k]) for i in range(len(m1[0])) ]) for j in range(len(m1)) for k in range(len(m2)) ]
print('matrix multiplier:')
debug('tensor_matrix(identity_matrix, identity_matrix)')
debug('tensor_matrix(_m2(1,0,0,1), _m2(4,5,6,7))')
debug('tensor_matrix(_m2(4,5,6,7), _m2(1,0,0,1))')
test('tensor_matrix(identity_matrix, identity_matrix) == [[(1+0j), 0j, 0j, 0j], [0j, (1+0j), 0j, 0j], [0j, 0j, (1+0j), 0j], [0j, 0j, 0j, (1+0j)]]')
test('tensor_matrix(_m2(1,0,0,1), _m2(4,5,6,7)) == [[(4+0j), (5+0j), 0j, 0j], [(6+0j), (7+0j), 0j, 0j], [0j, 0j, (4+0j), (5+0j)], [0j, 0j, (6+0j), (7+0j)]]')
test('tensor_matrix(_m2(4,5,6,7), _m2(1,0,0,1)) == [[(4+0j), 0j, (5+0j), 0j], [0j, (4+0j), 0j, (5+0j)], [(6+0j), 0j, (7+0j), 0j], [0j, (6+0j), 0j, (7+0j)]]')

print('multi-state-drifting')
state = tensor2(one, zero, zero)
debug('tensor2(one, zero, zero)')
debug('tensor_matrix(CNOT_matrix, identity_matrix)')
debug('_mul2(state, tensor_matrix(CNOT_matrix, identity_matrix))')
test('_mul2(tensor2(one, zero, zero), tensor_matrix(CNOT_matrix, identity_matrix)) == tensor2(one, one, zero)')
test('_mul2(tensor2(one, one, zero), tensor_matrix(CNOT_matrix, identity_matrix)) == tensor2(one, zero, zero)')
test('_mul2(tensor2(zero, zero, one), tensor_matrix(CNOT_matrix, identity_matrix)) == tensor2(zero, zero, one)')
test('_mul2(tensor2(zero, one, one), tensor_matrix(CNOT_matrix, identity_matrix)) == tensor2(zero, one, one)')
test('_mul2(tensor2(zero, zero, zero), tensor_matrix(identity_matrix, CNOT_matrix)) == tensor2(zero, zero, zero)')
test('_mul2(tensor2(zero, one, zero), tensor_matrix(identity_matrix, CNOT_matrix)) == tensor2(zero, one, one)')
test('_mul2(tensor2(zero, one, one), tensor_matrix(identity_matrix, CNOT_matrix)) == tensor2(zero, one, zero)')
test('_mul2(tensor2(zero, zero, one), tensor_matrix(identity_matrix, CNOT_matrix)) == tensor2(zero, zero, one)')


debug('_mul2(tensor2(zero, zero), tensor_matrix(HADAMARD_matrix, identity_matrix))')
test('_mul2(tensor2(zero, zero), tensor_matrix(HADAMARD_matrix, identity_matrix)) == tensor2(positive, zero)')
debug('_mul2(_mul2(tensor2(zero, zero), tensor_matrix(HADAMARD_matrix, identity_matrix)), CNOT_matrix)')
test('_mul2(_mul2(tensor2(zero, zero), tensor_matrix(HADAMARD_matrix, identity_matrix)), CNOT_matrix) == ((0.7071067811865475+0j), 0j, 0j, (0.7071067811865475+0j))')
debug('tensor2(positive, positive)')
test('_cmp2(tensor2(positive, positive), (0.5+0j,0.5+0j,0.5+0j,0.5+0j,))')


Z_matrix = _m2(
	1, 0,
	0, -1)
S_matrix = _m2(
	1, 0,
	0, complex(0,1))
T_matrix = _m2(
	1, 0,
	0, complex(sqrt1_2, sqrt1_2))
T_dagger_matrix = _m2(
	1, 0,
	0, complex(sqrt1_2, -sqrt1_2))
NOT_sqrt_matrix = _m2(
	complex(.5,.5), complex(.5,-.5),
	complex(.5,-.5), complex(.5,.5))
SWAP_sqrt_matrix = _m2(
	1,0,0,0,
	0,complex(.5,.5), complex(.5,-.5),0,
	0,complex(.5,-.5), complex(.5,.5),0,
	0,0,0,1)


print('more matrices')
debug('Z_matrix')
debug('S_matrix')
debug('T_matrix')
debug('T_dagger_matrix')
debug('NOT_sqrt_matrix')
debug('SWAP_sqrt_matrix')
debug('_mul2(one, NOTGATE_matrix)')
debug('_mul2(one, NOT_sqrt_matrix)')
debug('_mul2(_mul2(one, NOT_sqrt_matrix), NOT_sqrt_matrix)')
test('_mul2(one, NOTGATE_matrix) == zero')
test('_mul2(one, NOTGATE_matrix) == _mul2(_mul2(one, NOT_sqrt_matrix), NOT_sqrt_matrix)')
debug('_mul2(tensor2(one, zero), SWAP_matrix)')
debug('_mul2(tensor2(one, zero), SWAP_sqrt_matrix)')
debug('_mul2(_mul2(tensor2(one, zero), SWAP_sqrt_matrix), SWAP_sqrt_matrix)')
test('_mul2(tensor2(one, zero), SWAP_matrix) == _mul2(_mul2(tensor2(one, zero), SWAP_sqrt_matrix), SWAP_sqrt_matrix)')

_cmp_matrix = lambda z1,z2: all( cmath.isclose(z1[i],z2[i]) if type(z1[0]) != list and type(z1[0]) != tuple else _cmp_matrix(z1[i],z2[i]) for i in range(max(len(z1), len(z2))) )
def tensor_matrix2(m1,*ms):
	m = m1
	for om in ms:
		m = tensor_matrix(m, om)
	return m

debug('tensor_matrix(tensor_matrix(identity_matrix, identity_matrix), _m2(4,5,6,7))')
test('_cmp_matrix([[0,1]], [[0,1]]) == True')
test('_cmp_matrix([[0,1]], [[0,2]]) == False')
test('_cmp_matrix([[[0,1,1,1],[0,1,1,1],[0,1,-1,1j]]], [[[0,1,1,1],[0,1,1,1],[0,1,-1,1j]]]) == True')
test('_cmp_matrix([[[0,1,1,1],[0,1,1,1],[0,1,-1,0.5j]]], [[[0,1,1,1],[0,1,1,1],[0,1,-1,1j]]]) == False')
test('_cmp_matrix(tensor_matrix(tensor_matrix(identity_matrix, identity_matrix), _m2(4,5,6,7)), [[(4+0j), (5+0j), 0j, 0j, 0j, 0j, 0j, 0j], [(6+0j), (7+0j), 0j, 0j, 0j, 0j, 0j, 0j], [0j, 0j, (4+0j), (5+0j), 0j, 0j, 0j, 0j], [0j, 0j, (6+0j), (7+0j), 0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j, (4+0j), (5+0j), 0j, 0j], [0j, 0j, 0j, 0j, (6+0j), (7+0j), 0j, 0j], [0j, 0j, 0j, 0j, 0j, 0j, (4+0j), (5+0j)], [0j, 0j, 0j, 0j, 0j, 0j, (6+0j), (7+0j)]])')
test('_cmp_matrix(tensor_matrix2(identity_matrix, identity_matrix, _m2(4,5,6,7)), [[(4+0j), (5+0j), 0j, 0j, 0j, 0j, 0j, 0j], [(6+0j), (7+0j), 0j, 0j, 0j, 0j, 0j, 0j], [0j, 0j, (4+0j), (5+0j), 0j, 0j, 0j, 0j], [0j, 0j, (6+0j), (7+0j), 0j, 0j, 0j, 0j], [0j, 0j, 0j, 0j, (4+0j), (5+0j), 0j, 0j], [0j, 0j, 0j, 0j, (6+0j), (7+0j), 0j, 0j], [0j, 0j, 0j, 0j, 0j, 0j, (4+0j), (5+0j)], [0j, 0j, 0j, 0j, 0j, 0j, (6+0j), (7+0j)]])')
debug('tensor_matrix2(identity_matrix, SWAP_matrix)')
test('''_cmp_matrix(tensor_matrix2(identity_matrix, SWAP_matrix), [
	[(1+0j), 0j, 0j, 0j, 0j, 0j, 0j, 0j],
	[0j, 0j, (1+0j), 0j, 0j, 0j, 0j, 0j],
	[0j, (1+0j), 0j, 0j, 0j, 0j, 0j, 0j],
	[0j, 0j, 0j, (1+0j), 0j, 0j, 0j, 0j],
	[0j, 0j, 0j, 0j, (1+0j), 0j, 0j, 0j],
	[0j, 0j, 0j, 0j, 0j, 0j, (1+0j), 0j],
	[0j, 0j, 0j, 0j, 0j, (1+0j), 0j, 0j],
	[0j, 0j, 0j, 0j, 0j, 0j, 0j, (1+0j)]])''')
debug('_mul2(tensor2(zero, one, zero), tensor_matrix2(identity_matrix, identity_matrix, identity_matrix))')
test('_mul2(tensor2(zero, one, zero), tensor_matrix2(identity_matrix, identity_matrix, identity_matrix)) == tensor2(zero, one, zero)')
debug('_mul2(tensor2(zero, one, zero), tensor_matrix2(identity_matrix, SWAP_matrix))')
test('_mul2(tensor2(zero, one, zero), tensor_matrix2(identity_matrix, SWAP_matrix)) == tensor2(zero, zero, one)')

def compute_rotate_matrix(theta):
	return _m2(
		cos(theta/2), sin(theta/2) * complex(0,-1),
		sin(theta/2) * complex(0,-1), cos(theta/2))

debug('compute_rotate_matrix(0)')
debug('compute_rotate_matrix(math.pi)')
debug('range(len(compute_rotate_matrix(math.pi*4)))')
debug('_mul2(tensor2(zero), compute_rotate_matrix(math.pi*4))')
debug('(_mul2(tensor2(zero), compute_rotate_matrix(math.pi*4)), zero)')
test('_cmp2(_mul2(tensor2(zero), compute_rotate_matrix(math.pi)), ((6.123233995736766e-17+0j), -1j))')
test('_cmp2(_mul2(tensor2(zero), compute_rotate_matrix(math.pi*2)), ((-1+0j), -1.2246467991473532e-16j))')
test('_cmp2(_mul2(tensor2(zero), compute_rotate_matrix(math.pi*4)), ((1+0j), 2.4492935982947064e-16j))')


def str_tensor(z):
	size = math.log2(len(z))
	f = "{0:f}|{1:0" + str(int(size)) + "b}>"
	total = sum( sqrt(zi.real**2 + zi.imag**2) for zi in z )

	states = [ f.format(sqrt(z[i].real**2 + z[i].imag**2), i) for i in range(len(z)) if sqrt(z[i].real**2 + z[i].imag**2) > 0.0000000001 ]
	if len(states) == 1 and not abs(total - 1) > 0.0000000001:
		return ''.join([ ("|{1:0" + str(int(size)) + "b}>").format(sqrt(z[i].real**2 + z[i].imag**2), i) for i in range(len(z)) if sqrt(z[i].real**2 + z[i].imag**2) > 0.0000000001 ])
	return ' + '.join(states)

def tensor_from_string(s):
	qubits_map = {
		'0': zero,
		'1': one,
		'+': positive,
		'-': negative,
	}
	return tensor2(*[ qubits_map[c] for c in s[1:-1] ])


test('str_tensor(tensor2(zero)) == "|0>"')
test('str_tensor(tensor2(zero, one, zero)) == "|010>"')
test('str_tensor(tensor2(positive)) == "0.707107|0> + 0.707107|1>"')
test('str_tensor(_mul2(_mul2(tensor2(zero, zero), tensor_matrix2(HADAMARD_matrix, identity_matrix)), CNOT_matrix)) == "0.707107|00> + 0.707107|11>"')
test('str_tensor(tensor2(one, one)) == "|11>"')
debug('tensor_from_string("|000>")')
debug('tensor_from_string("|11>")')
debug('tensor_from_string("|+1>")')
debug('tensor_from_string("|+0>")')
test('str_tensor(tensor_from_string("|00>")) == "|00>"')
test('str_tensor(tensor_from_string("|10>")) == "|10>"')
test('str_tensor(tensor_from_string("|01>")) == "|01>"')
test('str_tensor(tensor_from_string("|11>")) == "|11>"')
debug('str_tensor(tensor_from_string("|+0>")) == "0.707107|00> + 0.707107|10>"')




Long_CNOT_matrix = _m2(
	1,0,0,0,0,0,0,0,
	0,1,0,0,0,0,0,0,
	0,0,1,0,0,0,0,0,
	0,0,0,1,0,0,0,0,
	0,0,0,0,0,1,0,0,
	0,0,0,0,1,0,0,0,
	0,0,0,0,0,0,0,1,
	0,0,0,0,0,0,1,0)


class Tensor(object):
	def __init__(self, *states):
		self._t = tensor2(*[ tensor_from_string(s) if type(s) == str else s for s in states ])
	def __str__(self):
		return str_tensor(self._t)
	def __mul__(self, other):
		if type(other) is Tensor:
			return Tensor(self._t, other._t)
		elif type(other) is MatrixGate:
			return Tensor(_mul2(self._t, other._t))
		elif type(other) is float or type(other) is int:
			return Tensor([ v * other for v in self._t ])
		else:
			raise Exception('invalid other:' + str(other))
	def __eq__(self, other):
		if type(other) is Tensor:
			if len(self._t) != len(other._t):
				raise Exception('invalid length tensor comparison: {} ?= {}'.format(str(self), str(other)))
			return all( cmath.isclose(self._t[i], other._t[i]) for i in range(len(self._t)) )
		elif type(other) is str:
			return str(self) == other
		else:
			raise Exception('invalid other:' + str(other))
	def size(self):
		return int(math.log2(len(self._t)))


class MultiTensor(object):
	def from_pattern(length):
		return MultiTensor(*[ ("|{0:0" + str(int(length)) + "b}>").format(i) for i in range(2 ** int(length)) ])
	def __init__(self, *tensors):
		self._t = [ Tensor(t) if type(t) == str else t for t in tensors ]
	def __str__(self):
		return ','.join([ str(t) for t in self._t ])
	def __mul__(self, other):
		if type(other) is Tensor or type(other) is MatrixGate or type(other) is float or type(other) is int:
			return MultiTensor(*[ t * other for t in self._t ])
		else:
			raise Exception('invalid other:' + str(other))
	def __eq__(self, other):
		if type(other) is MultiTensor:
			if len(self._t) != len(other._t):
				raise Exception('invalid length multi-tensor comparison: {} ?= {}'.format(str(self), str(other)))
			return all( self._t[i] == other._t[i] for i in range(len(self._t)) )
		elif type(other) is str:
			return str(self) == other
		else:
			raise Exception('invalid other:' + str(other))
	def size(self):
		return self._t[0].size()


def gate_from_string(s):
	gate_map = {
		'I': identity_matrix,
		'H': HADAMARD_matrix,
		'X': NOTGATE_matrix,
		'Z': Z_matrix,
		'S': S_matrix,
		'T': T_matrix,
		'Tdag': T_dagger_matrix,
		'CNOT': CNOT_matrix,
		'CCNOT': CCNOT_matrix,
		'SWAP': SWAP_matrix,
	}
	return tensor_matrix2(*[ gate_map[g.strip()] for g in s.split('*') ])
class MatrixGate(object):
	def __init__(self, *gates):
		self._t = tensor_matrix2(*[ gate_from_string(s) if type(s) == str else s for s in gates ])
	def __str__(self):
		return str(self._t)
	def __mul__(self, other):
		if type(other) is MatrixGate:
			return MatrixGate(self._t, other._t)
		else:
			raise Exception('invalid other:' + str(other))
	def __eq__(self, other):
		if type(other) is MatrixGate:
			return _cmp_matrix(self._t, other._t)
		elif type(other) is list:
			return _cmp_matrix(self._t, other)
		else:
			raise Exception('invalid other:' + str(other))
	def size(self):
		return int(math.log2(len(self._t)))
		


debug('Tensor("|00>", "|1>", "|0>")')
debug('Tensor("|0>", "|+>")')
debug('Tensor("|0>", "|1>") * Tensor("|0>", "|1>")')
debug('Tensor("|00>") * 2')

test('Tensor("|00>", "|1>", "|0>") == "|0010>"')
test('Tensor("|00>", "|1>", "|0>") != "|0110>"')
test('Tensor("|00>", "|1>", "|0>") == Tensor("|0010>")')
test('Tensor("|00>", "|1>", "|0>") != Tensor("|0110>")')
test('Tensor("|0>", "|1>", "|1>", "|0>") == Tensor("|0110>")')
test('Tensor("|0>", "|+>") == "0.707107|00> + 0.707107|01>"')
test('Tensor("|0>", "|1>") * Tensor("|0>", "|1>") == "|0101>"')
test('Tensor("|00>") * 2 == "2.000000|00>"')

debug('MatrixGate("I*I")')
debug('MatrixGate("I*X")')
test('MatrixGate("I*I") == [[(1+0j), 0j, 0j, 0j], [0j, (1+0j), 0j, 0j], [0j, 0j, (1+0j), 0j], [0j, 0j, 0j, (1+0j)]]')
test('MatrixGate("I*X") == [[0j, (1+0j), 0j, 0j], [(1+0j), 0j, 0j, 0j], [0j, 0j, 0j, (1+0j)], [0j, 0j, (1+0j), 0j]]')
test('MatrixGate("I*I") == tensor_matrix2(identity_matrix, identity_matrix)')
test('MatrixGate("I*I") != tensor_matrix2(identity_matrix, HADAMARD_matrix)')
test('MatrixGate("I*H") != tensor_matrix2(identity_matrix, identity_matrix)')
test('MatrixGate("I*H") == tensor_matrix2(identity_matrix, HADAMARD_matrix)')
test('MatrixGate("I*I") * MatrixGate("H") == tensor_matrix2(identity_matrix, identity_matrix, HADAMARD_matrix)')
test('MatrixGate("I*I") * MatrixGate("H") == MatrixGate("I*I*H")')
test('MatrixGate("I*I") * MatrixGate("H") != MatrixGate("I*I*H*I")')

debug('Tensor("|000>") * MatrixGate(Long_CNOT_matrix)')
debug('Tensor("|001>") * MatrixGate(Long_CNOT_matrix)')
debug('Tensor("|010>") * MatrixGate(Long_CNOT_matrix)')
debug('Tensor("|011>") * MatrixGate(Long_CNOT_matrix)')
debug('Tensor("|100>") * MatrixGate(Long_CNOT_matrix)')
debug('Tensor("|101>") * MatrixGate(Long_CNOT_matrix)')
debug('Tensor("|110>") * MatrixGate(Long_CNOT_matrix)')
debug('Tensor("|111>") * MatrixGate(Long_CNOT_matrix)')
debug('MultiTensor.from_pattern(3)')
debug('MultiTensor.from_pattern(3) * MatrixGate(Long_CNOT_matrix)')
# debug('''MultiTensor.from_pattern(3) * MatrixGate("I*I*H") * MatrixGate("I*CNOT")''')





Long_flip_CNOT_matrix = _m2(
	1,0,0,0,0,0,0,0,
	0,0,0,0,0,1,0,0,
	0,0,1,0,0,0,0,0,
	0,0,0,0,0,0,0,1,
	0,0,0,0,1,0,0,0,
	0,1,0,0,0,0,0,0,
	0,0,0,0,0,0,1,0,
	0,0,0,1,0,0,0,0)



Long_CNOT_matrix = _m2(
	1,0,0,0,0,0,0,0,
	0,1,0,0,0,0,0,0,
	0,0,1,0,0,0,0,0,
	0,0,0,1,0,0,0,0,
	0,0,0,0,0,1,0,0,
	0,0,0,0,1,0,0,0,
	0,0,0,0,0,0,0,1,
	0,0,0,0,0,0,1,0)

def expando_matrix(m):
	return _m2(
		m[0][0],m[0][1],0,0,m[0][2],m[0][3],0,0,
		m[1][0],m[1][1],0,0,m[1][2],m[1][3],0,0,
		0,0,m[0][0],m[0][1],0,0,m[0][2],m[0][3],
		0,0,m[1][0],m[1][1],0,0,m[1][2],m[1][3],
		m[2][0],m[2][1],0,0,m[2][2],m[2][3],0,0,
		m[3][0],m[3][1],0,0,m[3][2],m[3][3],0,0,
		0,0,m[2][0],m[2][1],0,0,m[2][2],m[2][3],
		0,0,m[3][0],m[3][1],0,0,m[3][2],m[3][3])

def expando_matrix2(m):
	ms = flattent(m)
	return _m2(
		*ms[4*0+0:1+4*0+1],0,0,*ms[4*0+2:1+4*0+3],0,0,
		*ms[4*1+0:1+4*1+1],0,0,*ms[4*1+2:1+4*1+3],0,0,
		0,0,*ms[4*0+0:1+4*0+1],0,0,*ms[4*0+2:1+4*0+3],
		0,0,*ms[4*1+0:1+4*1+1],0,0,*ms[4*1+2:1+4*1+3],
		*ms[4*2+0:1+4*2+1],0,0,*ms[4*2+2:1+4*2+3],0,0,
		*ms[4*3+0:1+4*3+1],0,0,*ms[4*3+2:1+4*3+3],0,0,
		0,0,*ms[4*2+0:1+4*2+1],0,0,*ms[4*2+2:1+4*2+3],
		0,0,*ms[4*3+0:1+4*3+1],0,0,*ms[4*3+2:1+4*3+3])

def concat_matrices(q0, q1, q2, q3):
	return [ q0[i] + q1[i] for i in range(len(q0)) ] + [ q2[i] + q3[i] for i in range(len(q0)) ]
def quad_matrix(m):
	length = len(m)
	halflength = int(length / 2)
	qs = []
	for i in range(4):
		start = (i % 2) * halflength
		end = (1 + i % 2) * halflength
		rangestart = int(i / 2) * halflength
		rangeend = int(1 + i / 2) * halflength
		q = [ m[n][start:end] for n in range(rangestart, rangeend) ]
		qs.append(q)
	return qs

def expando_matrix3(m):
	ms = flattent(m)
	q0 = tensor_matrix(identity_matrix, _m2(*ms[0:2], *ms[4:6]))
	q1 = tensor_matrix(identity_matrix, _m2(*ms[2:4], *ms[6:8]))
	q2 = tensor_matrix(identity_matrix, _m2(*ms[8:10], *ms[12:14]))
	q3 = tensor_matrix(identity_matrix, _m2(*ms[10:12], *ms[14:16]))

	return _m2(
		*q0[0], *q1[0],
		*q0[1], *q1[1],
		*q0[2], *q1[2],
		*q0[3], *q1[3],
		*q2[0], *q3[0],
		*q2[1], *q3[1],
		*q2[2], *q3[2],
		*q2[3], *q3[3])


def expando_matrix4(m):
	qs = [ tensor_matrix(identity_matrix, section) for section in quad_matrix(m) ]
	return concat_matrices(*qs)

def expando_matrix_times(m, times):
	for i in range(times):
		m = expando_matrix4(m)
	return m




test('MatrixGate(Long_CNOT_matrix) == MatrixGate(expando_matrix(CNOT_matrix))')
test('MatrixGate(Long_flip_CNOT_matrix) == MatrixGate(expando_matrix(flip_CNOT_matrix))')
debug('expando_matrix2(CNOT_matrix)')
test('MatrixGate(Long_CNOT_matrix) == MatrixGate(expando_matrix2(CNOT_matrix))')
test('MatrixGate(Long_flip_CNOT_matrix) == MatrixGate(expando_matrix2(flip_CNOT_matrix))')
debug('expando_matrix3(CNOT_matrix)')
test('MatrixGate(Long_CNOT_matrix) == MatrixGate(expando_matrix3(CNOT_matrix))')
test('MatrixGate(Long_flip_CNOT_matrix) == MatrixGate(expando_matrix3(flip_CNOT_matrix))')
debug('quad_matrix(_m2(1,2,3,4))')
debug('quad_matrix(flip_CNOT_matrix)')
debug('MultiTensor.from_pattern(3) * MatrixGate(expando_matrix3(CNOT_matrix))')
test('MultiTensor.from_pattern(3) * MatrixGate(expando_matrix(CNOT_matrix)) == "|000>,|001>,|010>,|011>,|101>,|100>,|111>,|110>"')
test('MultiTensor.from_pattern(3) * MatrixGate(expando_matrix2(CNOT_matrix)) == "|000>,|001>,|010>,|011>,|101>,|100>,|111>,|110>"')
test('MultiTensor.from_pattern(3) * MatrixGate(expando_matrix3(CNOT_matrix)) == "|000>,|001>,|010>,|011>,|101>,|100>,|111>,|110>"')
test('MultiTensor.from_pattern(3) * MatrixGate(expando_matrix4(CNOT_matrix)) == "|000>,|001>,|010>,|011>,|101>,|100>,|111>,|110>"')
debug('MultiTensor.from_pattern(3) * MatrixGate(expando_matrix3(flip_CNOT_matrix))')
test('MultiTensor.from_pattern(3) * MatrixGate(expando_matrix(flip_CNOT_matrix)) == "|000>,|101>,|010>,|111>,|100>,|001>,|110>,|011>"')
test('MultiTensor.from_pattern(3) * MatrixGate(expando_matrix2(flip_CNOT_matrix)) == "|000>,|101>,|010>,|111>,|100>,|001>,|110>,|011>"')
test('MultiTensor.from_pattern(3) * MatrixGate(expando_matrix3(flip_CNOT_matrix)) == "|000>,|101>,|010>,|111>,|100>,|001>,|110>,|011>"')
test('MultiTensor.from_pattern(3) * MatrixGate(expando_matrix4(flip_CNOT_matrix)) == "|000>,|101>,|010>,|111>,|100>,|001>,|110>,|011>"')
debug('expando_matrix4(expando_matrix4(CNOT_matrix))')
debug('MultiTensor.from_pattern(4) * MatrixGate(expando_matrix4(expando_matrix4(CNOT_matrix)))')
test('MultiTensor.from_pattern(4) * MatrixGate(expando_matrix4(expando_matrix4(CNOT_matrix))) == "|0000>,|0001>,|0010>,|0011>,|0100>,|0101>,|0110>,|0111>,|1001>,|1000>,|1011>,|1010>,|1101>,|1100>,|1111>,|1110>"')
debug('MultiTensor.from_pattern(4) * MatrixGate(expando_matrix4(expando_matrix4(flip_CNOT_matrix)))')
test('MultiTensor.from_pattern(4) * MatrixGate(expando_matrix4(expando_matrix4(flip_CNOT_matrix))) == "|0000>,|1001>,|0010>,|1011>,|0100>,|1101>,|0110>,|1111>,|1000>,|0001>,|1010>,|0011>,|1100>,|0101>,|1110>,|0111>"')
debug('MultiTensor.from_pattern(3) * MatrixGate(expando_matrix4(SWAP_matrix))')
test('MultiTensor.from_pattern(3) * MatrixGate(expando_matrix4(SWAP_matrix)) == "|000>,|100>,|010>,|110>,|001>,|101>,|011>,|111>"')
debug('MultiTensor.from_pattern(4) * MatrixGate(expando_matrix4(expando_matrix4(SWAP_matrix)))')
test('MultiTensor.from_pattern(4) * MatrixGate(expando_matrix4(expando_matrix4(SWAP_matrix))) == "|0000>,|1000>,|0010>,|1010>,|0100>,|1100>,|0110>,|1110>,|0001>,|1001>,|0011>,|1011>,|0101>,|1101>,|0111>,|1111>"')



def pad_gates(gate, gatelength, position, length):
	return ("I*" * (position - gatelength + 1)) + gate + ("*I" * (length - position - gatelength))
def pad_gates2(gate, position, length):
	gatelength = int(math.log2(len(gate)))
	return filter(lambda s: s != '', ['*'.join(['I'] * (position - gatelength + 1)), gate, '*'.join(['I'] * (length - position - gatelength + 1))])

def parse_instruction(inst):
	inst = inst.lower()
	if inst.startswith('state '):
		return lambda s: Tensor(inst[len('state '):])
	elif inst.startswith('not '):
		position = int(inst[len('not '):])
		return lambda s: s * MatrixGate(pad_gates("X", 1, s.size() - position - 1, s.size()))
	elif inst.startswith('hadamard '):
		position = int(inst[len('hadamard '):])
		return lambda s: s * MatrixGate(pad_gates("H", 1, s.size() - position - 1, s.size()))
	elif inst.startswith('tgate '):
		position = int(inst[len('tgate '):])
		return lambda s: s * MatrixGate(pad_gates("T", 1, s.size() - position - 1, s.size()))
	elif inst.startswith('tdagger '):
		position = int(inst[len('tdagger '):])
		return lambda s: s * MatrixGate(pad_gates("Tdag", 1, s.size() - position - 1, s.size()))
	elif inst.startswith('sgate '):
		position = int(inst[len('sgate '):])
		return lambda s: s * MatrixGate(pad_gates("S", 1, s.size() - position - 1, s.size()))
	elif inst.startswith('cnot '):
		(position_a, position_b) = map(int, inst[len('cnot '):].split(','))
		position = min(position_a, position_b)
		if position_a < position_b:
			gate = flip_CNOT_matrix
		elif position_a > position_b:
			gate = CNOT_matrix
		else:
			raise Exception("invalid command: " + inst)
		gate = expando_matrix_times(gate, abs(position_b - position_a) - 1)
		return lambda s: s * MatrixGate(*pad_gates2(gate, s.size() - position - 1, s.size()))
	elif inst.startswith('swap '):
		(position_a, position_b) = map(int, inst[len('swap '):].split(','))
		position = min(position_a, position_b)
		gate = SWAP_matrix
		if position_a == position_b:
			raise Exception("invalid command: " + inst)
		gate = expando_matrix_times(gate, abs(position_b - position_a) - 1)
		return lambda s: s * MatrixGate(*pad_gates2(gate, s.size() - position - 1, s.size()))
	else:
		raise Exception('invalid instruction:' + str(inst))


def parse_instructions_block(insts):
	print("compiling fun: {}".format(insts))
	execution = [ parse_instruction(inst) for inst in filter(lambda s: s != '', map(lambda s: s.strip(), insts.split('\n'))) ]
	def executable(s):
		for f in execution:
			s = f(s)
		return s
	return executable


fun = parse_instructions_block('''
not 0
cnot 1,0
cnot 2,0
''')
debug('fun(MultiTensor.from_pattern(3))')
test('fun(MultiTensor.from_pattern(3)) == "|001>,|000>,|010>,|011>,|100>,|101>,|111>,|110>"')

fun = parse_instructions_block('''
not 0
cnot 0,1
cnot 0,2
''')
debug('fun(MultiTensor.from_pattern(3))')
test('fun(MultiTensor.from_pattern(3)) == "|111>,|000>,|101>,|010>,|011>,|100>,|001>,|110>"')

fun = parse_instructions_block('''swap 0,2''')
debug('fun(MultiTensor.from_pattern(3))')
test('fun(MultiTensor.from_pattern(3)) == "|000>,|100>,|010>,|110>,|001>,|101>,|011>,|111>"')

fun = parse_instructions_block('''swap 0,3''')
debug('fun(MultiTensor.from_pattern(4))')
test('fun(MultiTensor.from_pattern(4)) == "|0000>,|1000>,|0010>,|1010>,|0100>,|1100>,|0110>,|1110>,|0001>,|1001>,|0011>,|1011>,|0101>,|1101>,|0111>,|1111>"')

fun = parse_instructions_block('''swap 0,4''')
debug('fun(MultiTensor.from_pattern(5))')
test('fun(MultiTensor.from_pattern(5)) == "|00000>,|10000>,|00010>,|10010>,|00100>,|10100>,|00110>,|10110>,|01000>,|11000>,|01010>,|11010>,|01100>,|11100>,|01110>,|11110>,|00001>,|10001>,|00011>,|10011>,|00101>,|10101>,|00111>,|10111>,|01001>,|11001>,|01011>,|11011>,|01101>,|11101>,|01111>,|11111>"')

fun = parse_instructions_block('''hadamard 0''')
debug('fun(MultiTensor.from_pattern(2))')
test('fun(MultiTensor.from_pattern(2)) == "0.707107|00> + 0.707107|01>,0.707107|00> + 0.707107|01>,0.707107|10> + 0.707107|11>,0.707107|10> + 0.707107|11>"')

# turns out this is just a very long and complicated ccnot gate :P
fun = parse_instructions_block('''
hadamard 2
cnot 1,2
tdagger 2
cnot 0,2
tgate 2
cnot 1,2
tdagger 2
cnot 0,2
tgate 2
hadamard 2
tdagger 1
cnot 0,1
tdagger 1
cnot 0,1
tgate 0
sgate 1
''')

for t in MultiTensor.from_pattern(3)._t:
	print(t, '->', fun(t))

