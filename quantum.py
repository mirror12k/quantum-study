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

_cmp_matrix = lambda z1,z2: all( cmath.isclose(z1[i],z2[i]) if type(z1[0]) != list and type(z1[0]) != tuple else _cmp_matrix(z1[i],z2[i]) for i in range(len(z1)) )
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
	states = [ f.format(z[i].real, i) for i in range(len(z)) if not cmath.isclose(z[i], 0) ]
	return ' + '.join(states)


test('str_tensor(tensor2(zero)) == "1.000000|0>"')
test('str_tensor(tensor2(zero, one, zero)) == "1.000000|010>"')
test('str_tensor(tensor2(positive)) == "0.707107|0> + 0.707107|1>"')
test('str_tensor(_mul2(_mul2(tensor2(zero, zero), tensor_matrix2(HADAMARD_matrix, identity_matrix)), CNOT_matrix)) == "0.707107|00> + 0.707107|11>"')
test('str_tensor(tensor2(one, one)) == "1.000000|11>"')

