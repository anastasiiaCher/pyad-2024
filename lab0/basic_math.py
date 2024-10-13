import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    ans = [[0]*(len(m2[0])) for _ in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m1[0])):
            ans[i][j] += m1[i][k]*m2[k][j]
    return ans


def functions(a_1, a_2):
    a1, b1, c1 = map(int,f.split())
    a2, b2, c2 = map(int,s.split())
    def f(x, a, b, c):
        return a*x**2+b*x+c
    print('Экстремум первой:', minimize_scalar(f, args = (a1,b1,c1)).x)
    print('Экстремум второй:', minimize_scalar(f, args = (a2,b2,c2)).x)
    ans = []
    for a, b, c in [(a1,b1,c1), (a2,b2,c2)]:
        ans.append([])
    D = b**2-4*a*c
    if a == 0 and b == 0 and c == 0:
        ans[-1].append(float('inf'))
    elif a == 0 and b == 0 and c != 0:
        pass
    elif a == 0 and b != 0:
        ans[-1].append(-c/b)
    else:
        if D < 0:
            ans[-1].append(complex(-b/(2*a), -abs(D**0.5)/(2*a)))
            ans[-1].append(complex(-b/(2*a), abs(D**0.5)/(2*a)))
        elif D == 0:
            ans[-1].append(-b/(2*a))
        elif D > 0:
            ans[-1].append((-b-D**0.5)/(2*a))
            ans[-1].append((-b+D**0.5)/(2*a))
        else:
            pass
    res = []
    if not ans[0] or not ans[1]:
        pass
    elif ans[0][0] == float('inf') and ans[1][0] == float('inf'):
        res.append(None)
    else:
        if ans[0][0] == float('inf'):
            res = ans[1]
        elif ans[1][0] == float('inf'):
            res = ans[0]
        else:
            for x in ans[0]:
                for y in ans[1]:
                    if x == y:
                        res.append(x)
    return res


def skew(x):
    sr = sum(a)/len(a)
    m3 = sum((x-sr)**3 for x in a)/len(a)
    sr = (sum((x-sr)**2 for x in a)/len(a))**0.5
    return round(m3/sr**3,2)


def kurtosis(x):
    sr = sum(a)/len(a)
    m4 = sum((x-sr)**4 for x in a)/len(a)
    sr = (sum((x-sr)**2 for x in a)/len(a))**0.5
    return round(m4/sr**4-3,2)
