def matrix_multiplication(matrix1, matrix2):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix1[0]) != len(matrix2):
        #перемножить нельзя
        raise ValueError
    else:
        result = []
        rows=len(matrix1)
        columns=len(matrix2)
        for i in range(rows):
            result.append([])
            for j in range(rows):
                result[i].append(0)
                for k in range(columns):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        return result
#print(matrix_multiplication([[1,2,3], [4,5,6]], [[1,2], [3,4], [5,6]]))

def functions(a_1, a_2):
    """
    Задание 2. На вход поступают две строки, содержащие коэффициенты двух функций.
    Необходимо определить, есть ли у графиков функций точки пересечения.
    Вернуть нужно координаты найденных точек списком, если они есть. None, если их бесконечно много.
    """
    inp=a_1.split()
    a1=int(inp[0])
    b1=int(inp[1])
    c1=int(inp[2])
    inp=a_2.split()
    a2=int(inp[0])
    b2=int(inp[1])
    c2=int(inp[2])
    if a1==a2 and b1==b2 and c1==c2:
        return None
    elif a1==0 and a2==0:
            if b1 == b2:
                return []
            else: 
                sol_x = (c2-c1)/(b1-b2)
                sol_y = sol_x * b1 + c1
                return [tuple([sol_x, sol_y])]
    else:
        a_t = a1-a2
        b_t=b1-b2
        c_t=c1-c2
        if a_t == 0:
            if b_t == 0:
                return []
            else: 
                sol_x = -c_t / b_t
                sol_y = sol_x *sol_x*a1 + sol_x*b1 + c1
                return [tuple([sol_x, sol_y])]
        else:
            D_t=b_t * b_t - 4 * a_t * c_t
            if D_t < 0:
                return []
            elif D_t == 0:
                sol_x = -b_t / 2 / a_t
                ol_y = sol_x *sol*x*a1 + sol_x*b1 + c1
                return [tuple([sol_x, sol_y])]
            else:
                sol1_x= (D_t ** 0.5 - b_t) / 2 / a_t
                sol1_y = sol1_x *sol1_x*a1 + sol1_x*b1 + c1
                sol2_x= (-D_t ** 0.5 - b_t) / 2 / a_t
                sol2_y = sol2_x *sol2_x*a1 + sol2_x*b1 + c1
                return [tuple([sol1_x, sol1_y]), tuple([sol2_x, sol2_y])]
"""
coeffs1 = "0 2 -1" 
coeffs2 = "1 -4 4"
print(functions(coeffs1, coeffs2))
"""
def ext(a_1):
    """
    Задание 2. На вход поступает строка, содержащие коэффициенты функции.
    Необходимо найти точки экстремума функции. 
    """
    inp=a_1.split()
    a1=int(inp[0])
    b1=int(inp[1])
    c1=int(inp[2])
    if a1 == 0:
        return None #у линейной функции экстремума нет
    else:
        x = -b1 / 2 /a1 #у квадратичной функции точка минимума - вершина
        y = a1 * x * x + b1 * x + c1
        return tuple([x, y])
#print(ext("1 2 1"))
def skew(arr):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    avg = sum(arr)/ len(arr)
    var = 0
    m3 = 0
    for i in arr:
        var += (i - avg) ** 2
        m3 += (i - avg) ** 3
    var /= len(arr)
    m3 /= len(arr)
    return round(m3 / var **(3/2), 2)
"""
import scipy.stats as sc
x1 = [2,3,2,5,7,2,2,8]
print(skew(x1))
print(round(sc.skew(x1), 2))
"""
def kurtosis(arr):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    # put your code here
    avg = sum(arr)/ len(arr)
    var = 0
    m4 = 0
    for i in arr:
        var += (i - avg) ** 2
        m4 += (i - avg) ** 4
    var /= len(arr)
    m4 /= len(arr)
    return round(m4 / var **2 - 3, 2)
"""
import scipy.stats as sc
x1 = [2,3,2,5,7,2,2,8]
print(kurtosis(x1))
print(round(sc.kurtosis(x1), 2))
"""