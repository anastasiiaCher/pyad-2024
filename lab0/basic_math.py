import numpy as np
import scipy as sc


def matrix_multiplication(m1, m2):
    f len(m1[0]) != len(m2):
        raise ValueError("Матрицы нельзя умножить из-за неправильных размеров")
    else:
        ans = [[0]*(len(m2[0])) for _ in range(len(m1))]
        for i in range(len(m1)):
            for j in range(len(m2[0])):
                for k in range(len(m1[0])):
                    ans[i][j] += m1[i][k]*m2[k][j]
        return ans


def functions(f, s):
    a1, b1, c1 = map(int,f.split())
    a2, b2, c2 = map(int,s.split())
    def f(x, a, b, c):
        return a*x**2+b*x+c
    print('Экстремум первой:', minimize_scalar(f, args = (a1,b1,c1)).x)
    print('Экстремум второй:', minimize_scalar(f, args = (a2,b2,c2)).x)
    a, b, c = a1-a2, b1-b2, c1-c2
    ans = []
    D = b**2-4*a*c
    if a == 0 and b == 0 and c == 0:
        ans.append(float('inf'))
    elif a == 0 and b == 0 and c != 0:
        pass
    elif a == 0 and b != 0:
        ans.append(-c/b)
    else:
        if D < 0:
            ans.append(complex(-b/(2*a), -abs(D**0.5)/(2*a)))
            ans.append(complex(-b/(2*a), abs(D**0.5)/(2*a)))
        elif D == 0:
            ans.append(-b/(2*a))
        elif D > 0:
            ans.append((-b-D**0.5)/(2*a))
            ans.append((-b+D**0.5)/(2*a))
        else:
            pass
    res = []
    if not ans:
        pass
    elif ans[0] == float('inf'):
        res = None
    else:
        for x in ans:
            res.append((x,f(x,a1,b1,c1)))
        res.sort()
    return res


def skew(a):
    sr = sum(a)/len(a)
    m3 = sum((x-sr)**3 for x in a)/len(a)
    sr = (sum((x-sr)**2 for x in a)/len(a))**0.5
    return round(m3/sr**3,2)


def kurtosis(a):
    sr = sum(a)/len(a)
    m4 = sum((x-sr)**4 for x in a)/len(a)
    sr = (sum((x-sr)**2 for x in a)/len(a))**0.5
    return round(m4/sr**4-3,2)
