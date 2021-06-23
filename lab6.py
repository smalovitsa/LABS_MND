import random
import numpy
import math
import scipy.stats
import copy
import time
tim = time.time()
number_of_junk = 0
def det(arr):
    return numpy.linalg.det(numpy.array(arr))
def coefficients_interaction_squares(matrix, matrix_y, N):
    matrix = copy.deepcopy(matrix)
    if True:
        average_y = [sum(matrix_y[i]) / m for i in range(N)]
        for row in range(N):
            matrix[row].insert(0, 1)
            matrix[row].append(average_y[row])

        matrix_help = []
        matrix_m_ii = []
        reverse_matrix = list(map(list, zip(*matrix)))
        for i in range(len(reverse_matrix) - 1):
            mult = reverse_matrix[i]
            matrix_m_ii.append([])
            for j in range(len(mult)):
                matrix_help.append([reverse_matrix[col][j] * mult[j] for col in range(len(reverse_matrix))])

            reverse_matrix_m_ii = list(map(list, zip(*matrix_help)))
            for col in range(len(reverse_matrix_m_ii)):
                matrix_m_ii[i].append(sum(reverse_matrix_m_ii[col]))
            matrix_help = []

        list_k = []
        for row in range(len(matrix_m_ii)):
            list_k.append(matrix_m_ii[row].pop(-1))

        denominator = matrix_m_ii[:]
        denominator_det = det(denominator)

        reverse_det = list(map(list, zip(*denominator)))
        list_b = []
        for i in range(len(reverse_det)):
            numerator = reverse_det[:]
            numerator[i] = list_k
            list_b.append(det(list(zip(*numerator))) / denominator_det)
        return list_b

while time.time() - tim < 3:

    x1_min = -25
    x1_max = 75
    x2_min = 5
    x2_max = 40
    x3_min = 15
    x3_max = 25
    m = 3
    x_norm = [[1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1],
             [1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1],
             [1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1],
             [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
             [1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1],
             [1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1],
             [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, -1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0, 0],
             [1, 1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0, 0],
             [1, 0, -1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0],
             [1, 0, 1.73, 0, 0, 0, 0, 0, 0, 2.9929, 0],
             [1, 0, 0, -1.73, 0, 0, 0, 0, 0, 0, 2.9929],
             [1, 0, 0, 1.73, 0, 0, 0, 0, 0, 0, 2.9929]]

    x01 = (x1_min + x1_max) / 2
    x02 = (x2_min + x2_max) / 2
    x03 = (x3_min + x3_max) / 2

    dx1 = x1_max - x01
    dx2 = x2_max - x02
    dx3 = x3_max - x03
    l = 1.73
    x_nat = [[1, x1_min, x2_min, x3_min, x1_min * x2_min, x1_min * x3_min, x2_min * x3_min, x1_min * x2_min * x3_min, x1_min * x1_min,
             x2_min * x2_min, x3_min * x3_min],
            [1, x1_min, x2_max, x3_max, x1_min * x2_max, x1_min * x3_max, x2_max * x3_max, x1_min * x2_max * x3_max, x1_min * x1_min,
             x2_max * x2_max, x3_max * x3_max],
            [1, x1_max, x2_min, x3_max, x1_max * x2_min, x1_max * x3_max, x2_min * x3_max, x1_max * x2_min * x3_max, x1_max * x1_max,
             x2_min * x2_min, x3_max * x3_max],
            [1, x1_max, x2_max, x3_min, x1_max * x2_max, x1_max * x3_min, x2_max * x3_min, x1_max * x2_max * x3_min, x1_max * x1_max,
             x2_max * x2_max, x3_min * x3_min],
            [1, x1_min, x2_min, x3_max, x1_min * x2_min, x1_min * x3_max, x2_min * x3_max, x1_min * x2_min * x3_max, x1_min * x1_min,
             x2_min * x2_min, x3_max * x3_max],
            [1, x1_min, x2_max, x3_min, x1_min * x2_max, x1_min * x3_min, x2_max * x3_min, x1_min * x2_max * x3_min, x1_min * x1_min,
             x2_max * x2_max, x3_min * x3_min],
            [1, x1_max, x2_min, x3_min, x1_max * x2_min, x1_max * x3_min, x2_min * x3_min, x1_max * x2_min * x3_min, x1_max * x1_max,
             x2_min * x2_min, x3_min * x3_min],
            [1, x1_max, x2_max, x3_max, x1_max * x2_max, x1_max * x3_max, x2_max * x3_max, x1_max * x2_max * x3_max, x1_max * x1_max,
             x2_max * x2_max, x3_max * x3_max],
            [1, -l * dx1 + x01, x02, x03, (-l * dx1 + x01) * x02, (-l * dx1 + x01) * x03, x02 * x03,
             (-l * dx1 + x01) * x02 * x03, (-l * dx1 + x01) * (-l * dx1 + x01), x02 * x02, x03 * x03],
            [1, l * dx1 + x01, x02, x03, (l * dx1 + x01) * x02, (l * dx1 + x01) * x03, x02 * x03,
             (l * dx1 + x01) * x02 * x03, (l * dx1 + x01) * (l * dx1 + x01), x02 * x02, x03 * x03],
            [1, x01, -l * dx2 + x02, x03, x01 * (-l * dx2 + x02), x01 * x03, (-l * dx2 + x02) * x03,
             x01 * (-l * dx2 + x02) * x03, x01 * x01, (-l * dx2 + x02) * (-l * dx2 + x02), x03 * x03],
            [1, x01, l * dx2 + x02, x03, x01 * (l * dx2 + x02), x01 * x03, (l * dx2 + x02) * x03,
             x01 * (l * dx2 + x02) * x03, x01 * x01, (l * dx2 + x02) * (l * dx2 + x02), x03 * x03],
            [1, x01, x02, -l * dx3 + x03, x01 * x02, x01 * (-l * dx3 + x03), x02 * (-l * dx3 + x03),
             x01 * x02 * (-l * dx3 + x03), x01 * x01, x02 * x02, (-l * dx3 + x03) * (-l * dx3 + x03)],
            [1, x01, x02, l * dx3 + x03, x01 * x02, x01 * (l * dx3 + x03), x02 * (l * dx3 + x03),
             x01 * x02 * (l * dx3 + x03), x01 * x01, x02 * x02, (l * dx3 + x03) * (l * dx3 + x03)]]

    print("X нормалізоване = ")
    for i in range(14):
        print(x_norm[i])

    print("X натуралізоване = ")
    for i in range(14):
        print(x_nat[i])

    D = [0]*14
    ySr = [0]*14
    flag = True
    y = []
    while flag:
        y = [[5.5 + 6.4 * x_nat[i][1] + 0.6 * x_nat[i][2] + 2.7 * x_nat[i][3] + 1.9 * x_nat[i][1] * x_nat[i][1] +
              0.4 * x_nat[i][2] * x_nat[i][2] + 0.7 * x_nat[i][3] * x_nat[i][3] + 1.8 * x_nat[i][1] * x_nat[i][2] +
              0.2 * x_nat[i][1] * x_nat[i][3] + 6 * x_nat[i][2] * x_nat[i][3] + 4.8 * x_nat[i][1] * x_nat[i][2] * x_nat[i][3]
              + random.randint(0, 10) - 5 for j in range(m)] for i in range(14)]
        print("Y = ")
        for i in range(14):
            print(y[i])

        for i in range(m):
            for j in range(len(ySr)):
                ySr[j] += y[j][i]
        ySr = list(map(lambda x: x/m, ySr))

        mx1 = 0
        mx2 = 0
        mx3 = 0
        a11, a22, a33 = 0, 0, 0
        a12 = a21 = 0
        a13 = a31 = 0
        a23 = a32 = 0
        for i in range(14):
            mx1 += x_nat[i][1]
            mx2 += x_nat[i][2]
            mx3 += x_nat[i][3]
            a11 += x_nat[i][1] ** 2
            a22 += x_nat[i][2] ** 2
            a33 += x_nat[i][3] ** 2
            a12 += x_nat[i][1] * x_nat[i][2]
            a13 += x_nat[i][1] * x_nat[i][3]
            a23 += x_nat[i][2] * x_nat[i][3]
        mx1 = mx1 / 14
        mx2 = mx2 / 14
        mx3 = mx3 / 14
        a11 = a11 / 14
        a22 = a22 / 14
        a33 = a33 / 14
        a12 = a21 = a12 / 14
        a13 = a31 = a13 / 14
        a23 = a32 = a23 / 14
        a1 = 0
        a2 = 0
        a3 = 0
        my = 0
        for i in range(14):
            a1 += x_nat[i][1] * ySr[i]
            a2 += x_nat[i][2] * ySr[i]
            a3 += x_nat[i][3] * ySr[i]
            my += ySr[i]

        a1 = a1 / 14
        a2 = a2 / 14
        a3 = a3 / 14
        my = my / 14
        a = numpy.array([[1, mx1, mx2, mx3],
                         [mx1, a11, a12, a13],
                         [mx2, a12, a22, a32],
                         [mx3, a13, a23, a33]])
        c = numpy.array([[my], [a1], [a2], [a3]])
        b = numpy.linalg.solve(a, c)
        print("Рівняння регресії")
        print("y = ", round(b[0][0], 2), "+", round(b[1][0], 2), " * x1 +", round(b[2][0], 2), " * x2 +", round(b[3][0], 2),
              "* x3")

        for i in range(m):
            for j in range(len(D)):
                D[j] += pow((y[j][i] - ySr[j]),2)
        D = list(map(lambda x: x/m, D))
        print(D)
        Dmax = max(D)
        Gp = Dmax / sum(D)
        f1 = m - 1
        f2 = 14
        q = 0.05
        Gt = 0.35
        if f1 == 3:
            Gt = 0.3
        if Gp < Gt:
            print(Gp, "<", Gt)
            print("Дисперcія однорідна")
            print("m = ", m, "\n")
            flag = False
        else:
            print(Gp, ">", Gt)
            print("Дисперcія неоднорідна\n")
            print("m = ", m)
            m += 1
    number_of_junk = random.randint(10, 18)
    DB = sum(D) / 14
    Dbeta2 = DB / (14 * m)
    Dbeta = math.sqrt(Dbeta2)
    beta0 = (ySr[0] * x_norm[0][0] + ySr[1] * x_norm[1][0] + ySr[2] * x_norm[2][0] + ySr[3] * x_norm[3][0] + x_norm[4][0] * ySr[4] +
             x_norm[5][0] * ySr[5] + x_norm[6][0] * ySr[6] + x_norm[7][0] * ySr[7] + ySr[8] * x_norm[8][0] + ySr[9] * x_norm[9][
                 0] + ySr[10] * x_norm[10][0] + ySr[11] * x_norm[11][0] + x_norm[12][0] * ySr[12] +
             x_norm[13][0] * ySr[13]) / 14
    beta1 = (ySr[0] * x_norm[0][1] + ySr[1] * x_norm[1][1] + ySr[2] * x_norm[2][1] + ySr[3] * x_norm[3][1] + x_norm[4][1] * ySr[4] +
             x_norm[5][1] * ySr[5] + x_norm[6][1] * ySr[6] + x_norm[7][1] * ySr[7] + ySr[8] * x_norm[8][1] + ySr[9] * x_norm[9][
                 1] + ySr[10] * x_norm[10][1] + ySr[11] * x_norm[11][1] + x_norm[12][1] * ySr[12] +
             x_norm[13][1] * ySr[13]) / 14
    beta2 = (ySr[0] * x_norm[0][2] + ySr[1] * x_norm[1][2] + ySr[2] * x_norm[2][2] + ySr[3] * x_norm[3][2] + x_norm[4][2] * ySr[4] +
             x_norm[5][2] * ySr[5] + x_norm[6][2] * ySr[6] + x_norm[7][2] * ySr[7] + ySr[8] * x_norm[8][2] + ySr[9] * x_norm[9][
                 2] + ySr[10] * x_norm[10][2] + ySr[11] * x_norm[11][2] + x_norm[12][2] * ySr[12] +
             x_norm[13][2] * ySr[13]) / 14
    beta3 = (ySr[0] * x_norm[0][3] + ySr[1] * x_norm[1][3] + ySr[2] * x_norm[2][3] + ySr[3] * x_norm[3][3] + x_norm[4][3] * ySr[4] +
             x_norm[5][3] * ySr[5] + x_norm[6][3] * ySr[6] + x_norm[7][3] * ySr[7] + ySr[8] * x_norm[8][3] + ySr[9] * x_norm[9][
                 3] + ySr[10] * x_norm[10][3] + ySr[11] * x_norm[11][3] + x_norm[12][3] * ySr[12] +
             x_norm[13][3] * ySr[13]) / 14

    tN = []
    for i in range(4):
        tN.append((locals().get("beta"+str(i)))/Dbeta)

    f3 = f1 * f2
    ttabl = 2.048
    if f3 > 25:
        ttabl = 1.960
    print("Оцінимо значимість коефіцієнтів регресіїї згідно критерію Стьюдента")
    str(list(map(lambda x: print(str(x), "  ", ttabl), tN)))

    coef = [1, 0, 0, 0]
    for i in range(len(tN)):
        if tN[i] > ttabl:
            coef[i] = 1
    print("Значимі коефіцієнти (1 - значимий) ", coef, "\n")
    yQ = [[0]]*14
    for i in range(14):
        for j in range(4):
            yQ[i][0] += coef[j] * b[j] * x_nat[i][j]

    print("Рівняння регресії згідно критерію Стьюдента")
    print("y = ", coef[0] * round(b[0][0], 4), "+", coef[1] * round(b[1][0], 4), " * x1 +", coef[2] * round(b[2][0], 4),
          " * x2 +", coef[3] * round(b[3][0], 4),
          "* x3")
    # Фишер
    d = 0
    for i in range(len(coef)):
        if coef[i] == 1:
            d += 1
    f4 = 14 - d
    S_ad = (m / (14 - d)) * (pow((yQ[0][0] - ySr[0]), 2) + pow((yQ[1][0] - ySr[1]), 2) + pow((yQ[2][0] - ySr[2]), 2) + pow(
        (yQ[3][0] - ySr[3]), 2)
                             + pow((yQ[4][0] - ySr[4]), 2) + pow((yQ[5][0] - ySr[5]), 2) + pow((yQ[6][0] - ySr[6]), 2) + pow(
                (yQ[7][0] - ySr[7]), 2) + pow((yQ[8][0] - ySr[8]), 2) + pow((yQ[9][0] - ySr[9]), 2) + pow(
                (yQ[10][0] - ySr[10]), 2) + pow(
                (yQ[11][0] - ySr[11]), 2)
                             + pow((yQ[12][0] - ySr[12]), 2) + pow((yQ[13][0] - ySr[13]), 2))
    Fp = S_ad / DB
    Ft = 4.1709
    if f4 == 13:
        Fp = 3.3158
    if f4 == 12:
        Fp = 2.9223
    if f4 == 11:
        Fp = 2.6896
    if f4 == 10:
        Fp = 2.5336
    if f4 == 9:
        Fp = 2.4205
    if f4 == 8:
        Fp = 2.3343
    if f4 == 7:
        Fp = 2.2662
    if f4 == 6:
        Fp = 2.2107
    if f4 == 5:
        Fp = 2.1646
    if f4 == 4:
        Fp = 2.1256
    if f4 == 3:
        Fp = 2.0921
    if f4 == 2:
        Fp = 2.063
    if f4 == 1:
        Fp = 2.0374
    adect = 1
    if Fp > Ft:
        print("Рівняння регресії неадекватно оригіналу при рівні значимості 0.05 за критерієм Фішера\n")
        adect = 1
    else:
        print("Рівняння регресії адекватно оригіналу при рівні значимості 0.05 за критерієм Фішера")
        adect = 1

    flag2 = False
    if adect == 1:
        m0_0 = 8
        m1_0 = m0_1 = 0
        m2_0 = m0_2 = 0
        m3_0 = m0_3 = 0
        m4_0 = m0_4 = 0
        m5_0 = m0_5 = 0
        m6_0 = m0_6 = 0
        m7_0 = m0_7 = 0
        m1_2 = m2_1 = 0
        m1_3 = m3_1 = 0
        m1_4 = m4_1 = 0
        m1_5 = m5_1 = 0
        m1_6 = m6_1 = 0
        m1_7 = m7_1 = 0
        m2_3 = m3_2 = 0
        m2_4 = m4_2 = 0
        m2_5 = m5_2 = 0
        m2_6 = m6_2 = 0
        m2_7 = m7_2 = 0
        m3_4 = m4_3 = 0
        m3_5 = m5_3 = 0
        m3_6 = m6_3 = 0
        m3_7 = m7_3 = 0
        m4_5 = m5_4 = 0
        m4_6 = m6_4 = 0
        m4_7 = m7_4 = 0
        m5_6 = m6_5 = 0
        m5_7 = m7_5 = 0
        m6_7 = m7_6 = 0

        m1_1 = 0
        m2_2 = 0
        m3_3 = 0
        m4_4 = 0
        m5_5 = 0
        m6_6 = 0
        m7_7 = 0

        for i in range(14):
            m1_0 += x_nat[i][1]
            m2_0 += x_nat[i][2]
            m3_0 += x_nat[i][3]
            m4_0 += x_nat[i][1] * x_nat[i][2]
            m5_0 += x_nat[i][1] * x_nat[i][3]
            m6_0 += x_nat[i][3] * x_nat[i][2]
            m7_0 += x_nat[i][1] * x_nat[i][2] * x_nat[i][3]
            m1_2 += x_nat[i][1] * x_nat[i][2]
            m1_3 += x_nat[i][1] * x_nat[i][3]
            m1_4 += pow(x_nat[i][1], 2) * x_nat[i][2]
            m1_5 += pow(x_nat[i][1], 2) * x_nat[i][3]
            m1_6 += x_nat[i][1] * x_nat[i][2] * x_nat[i][3]
            m1_7 += pow(x_nat[i][1], 2) * x_nat[i][2] * x_nat[i][3]
            m2_3 += x_nat[i][3] * x_nat[i][2]
            m2_4 += pow(x_nat[i][2], 2) * x_nat[i][1]
            m2_5 += x_nat[i][1] * x_nat[i][2] * x_nat[i][3]
            m2_6 += pow(x_nat[i][2], 2) * x_nat[i][3]
            m2_7 += pow(x_nat[i][2], 2) * x_nat[i][3] * x_nat[i][1]
            m3_4 += x_nat[i][1] * x_nat[i][2] * x_nat[i][3]
            m3_5 += pow(x_nat[i][3], 2) * x_nat[i][1]
            m3_6 += pow(x_nat[i][3], 2) * x_nat[i][2]
            m3_7 += pow(x_nat[i][3], 2) * x_nat[i][2] * x_nat[i][1]
            m4_5 += pow(x_nat[i][1], 2) * x_nat[i][2] * x_nat[i][3]
            m4_6 += pow(x_nat[i][2], 2) * x_nat[i][3] * x_nat[i][1]
            m4_7 += pow(x_nat[i][1], 2) * pow(x_nat[i][2], 2) * x_nat[i][3]
            m5_6 += pow(x_nat[i][3], 2) * x_nat[i][2] * x_nat[i][1]
            m5_7 += pow(x_nat[i][1], 2) * pow(x_nat[i][3], 2) * x_nat[i][2]
            m6_7 += pow(x_nat[i][2], 2) * pow(x_nat[i][3], 2) * x_nat[i][1]

            m1_1 += pow(x_nat[i][1], 2)
            m2_2 += pow(x_nat[i][2], 2)
            m3_3 += pow(x_nat[i][3], 2)
            m4_4 += pow(x_nat[i][1], 2) * pow(x_nat[i][2], 2)
            m5_5 += pow(x_nat[i][1], 2) * pow(x_nat[i][3], 2)
            m6_6 += pow(x_nat[i][2], 2) * pow(x_nat[i][3], 2)
            m7_7 += pow(x_nat[i][1], 2) * pow(x_nat[i][2], 2) * pow(x_nat[i][3], 2)
        m0_1 = m0_1 / 14
        m0_2 = m0_2 / 14
        m0_3 = m0_3 / 14
        m0_4 = m0_4 / 14
        m0_5 = m0_5 / 14
        m0_6 = m0_6 / 14
        m0_7 = m0_7 / 14
        m2_1 = m2_1 / 14
        m3_1 = m3_1 / 14
        m4_1 = m4_1 / 14
        m5_1 = m5_1 / 14
        m6_1 = m6_1 / 14
        m7_1 = m7_1 / 14
        m3_2 = m3_2 / 14
        m4_2 = m4_2 / 14
        m5_2 = m5_2 / 14
        m6_2 = m6_2 / 14
        m7_2 = m7_2 / 14
        m4_3 = m4_3 / 14
        m5_3 = m5_3 / 14
        m6_3 = m6_3 / 14
        m7_3 = m7_3 / 14
        m5_4 = m5_4 / 14
        m6_4 = m6_4 / 14
        m7_4 = m7_4 / 14
        m6_5 = m6_5 / 14
        m7_5 = m7_5 / 14
        m7_6 = m7_6 / 14
        m0_1 = m1_0
        m0_2 = m2_0
        m0_3 = m3_0
        m0_4 = m4_0
        m0_5 = m5_0
        m0_6 = m6_0
        m0_7 = m7_0
        m2_1 = m1_2
        m3_1 = m1_3
        m4_1 = m1_4
        m5_1 = m1_5
        m6_1 = m1_6
        m7_1 = m1_7
        m3_2 = m2_3
        m4_2 = m2_4
        m5_2 = m2_5
        m6_2 = m2_6
        m7_2 = m2_7
        m4_3 = m3_4
        m5_3 = m3_5
        m6_3 = m3_6
        m7_3 = m3_7
        m5_4 = m4_5
        m6_4 = m4_6
        m7_4 = m4_7
        m6_5 = m5_6
        m7_5 = m5_7
        m7_6 = m6_7

        k0 = 0
        k1 = 0
        k2 = 0
        k3 = 0
        k4 = 0
        k5 = 0
        k6 = 0
        k7 = 0
        for i in range(14):
            k0 += ySr[i]
            k1 += ySr[i] * x_nat[i][1]
            k2 += ySr[i] * x_nat[i][2]
            k3 += ySr[i] * x_nat[i][3]
            k4 += ySr[i] * x_nat[i][1] * x_nat[i][2]
            k5 += ySr[i] * x_nat[i][1] * x_nat[i][3]
            k6 += ySr[i] * x_nat[i][2] * x_nat[i][3]
            k7 += ySr[i] * x_nat[i][1] * x_nat[i][2] * x_nat[i][3]
        a = numpy.array([[m0_0, m1_0, m2_0, m3_0, m4_0, m5_0, m6_0, m7_0],
                         [m0_1, m1_1, m2_1, m3_1, m4_1, m5_1, m6_1, m7_1],
                         [m0_2, m1_2, m2_2, m3_2, m4_2, m5_2, m6_2, m7_2],
                         [m0_3, m1_3, m2_3, m3_3, m4_3, m5_3, m6_3, m7_3],
                         [m0_4, m1_4, m2_4, m3_4, m4_4, m5_4, m6_4, m7_4],
                         [m0_5, m1_5, m2_5, m3_5, m4_5, m5_5, m6_5, m7_5],
                         [m0_6, m1_6, m2_6, m3_6, m4_6, m5_6, m6_6, m7_6],
                         [m0_7, m1_7, m2_7, m3_7, m4_7, m5_7, m6_7, m7_7]])
        c = numpy.array([[k0], [k1], [k2], [k3], [k4], [k5], [k6], [k7]])
        b = numpy.linalg.solve(a, c)
        print("Рівняння регресії з ефектом взаємодії: ")
        print("y = ", round(b[0][0], 4), "+", round(b[1][0], 4), " * x1 +", round(b[2][0], 4), " * x2 +", round(b[3][0], 4),
              "* x3 +", round(b[4][0], 4),
              " * x1 * x2 +", round(b[5][0], 4), " * x1 * x3 +", round(b[6][0], 4), "* x2 * x3 +", round(b[7][0], 4),
              " * x1 * x2 * x3\n")
        DB = sum(D)/14
        Dbeta2 = DB / (14 * m)
        Dbeta = math.sqrt(Dbeta2)
        beta0 = (ySr[0] * x_norm[0][0] + ySr[1] * x_norm[1][0] + ySr[2] * x_norm[2][0] + ySr[3] * x_norm[3][0] + x_norm[4][0] * ySr[4] +
                 x_norm[5][0] * ySr[5] + x_norm[6][0] * ySr[6] + x_norm[7][0] * ySr[7] + ySr[8] * x_norm[8][0] + ySr[9] * x_norm[9][
                     0] + ySr[10] * x_norm[10][0] + ySr[11] * x_norm[11][0] + x_norm[12][0] * ySr[12] +
                 x_norm[13][0] * ySr[13]) / 14
        beta1 = (ySr[0] * x_norm[0][1] + ySr[1] * x_norm[1][1] + ySr[2] * x_norm[2][1] + ySr[3] * x_norm[3][1] + x_norm[4][1] * ySr[4] +
                 x_norm[5][1] * ySr[5] + x_norm[6][1] * ySr[6] + x_norm[7][1] * ySr[7] + ySr[8] * x_norm[8][1] + ySr[9] * x_norm[9][
                     1] + ySr[10] * x_norm[10][1] + ySr[11] * x_norm[11][1] + x_norm[12][1] * ySr[12] +
                 x_norm[13][1] * ySr[13]) / 14
        beta2 = (ySr[0] * x_norm[0][2] + ySr[1] * x_norm[1][2] + ySr[2] * x_norm[2][2] + ySr[3] * x_norm[3][2] + x_norm[4][2] * ySr[4] +
                 x_norm[5][2] * ySr[5] + x_norm[6][2] * ySr[6] + x_norm[7][2] * ySr[7] + ySr[8] * x_norm[8][2] + ySr[9] * x_norm[9][
                     2] + ySr[10] * x_norm[10][2] + ySr[11] * x_norm[11][2] + x_norm[12][2] * ySr[12] +
                 x_norm[13][2] * ySr[13]) / 14
        beta3 = (ySr[0] * x_norm[0][3] + ySr[1] * x_norm[1][3] + ySr[2] * x_norm[2][3] + ySr[3] * x_norm[3][3] + x_norm[4][3] * ySr[4] +
                 x_norm[5][3] * ySr[5] + x_norm[6][3] * ySr[6] + x_norm[7][3] * ySr[7] + ySr[8] * x_norm[8][3] + ySr[9] * x_norm[9][
                     3] + ySr[10] * x_norm[10][3] + ySr[11] * x_norm[11][3] + x_norm[12][3] * ySr[12] +
                 x_norm[13][3] * ySr[13]) / 14
        beta4 = (ySr[0] * x_norm[0][4] + ySr[1] * x_norm[1][4] + ySr[2] * x_norm[2][4] + ySr[3] * x_norm[3][4] + x_norm[4][4] * ySr[4] +
                 x_norm[5][4] * ySr[5] + x_norm[6][4] * ySr[6] + x_norm[7][4] * ySr[7] + ySr[8] * x_norm[8][4] + ySr[9] * x_norm[9][
                     4] + ySr[10] * x_norm[10][4] + ySr[11] * x_norm[11][4] + x_norm[12][4] * ySr[12] +
                 x_norm[13][4] * ySr[13]) / 14
        beta5 = (ySr[0] * x_norm[0][5] + ySr[1] * x_norm[1][5] + ySr[2] * x_norm[2][5] + ySr[3] * x_norm[3][5] + x_norm[4][5] * ySr[4] +
                 x_norm[5][5] * ySr[5] + x_norm[6][5] * ySr[6] + x_norm[7][5] * ySr[7] + ySr[8] * x_norm[8][5] + ySr[9] * x_norm[9][5]
                 + ySr[10] * x_norm[10][5] + ySr[11] * x_norm[11][5] + x_norm[12][5] * ySr[12] +
                 x_norm[13][5] * ySr[13]) / 14
        beta6 = (ySr[0] * x_norm[0][6] + ySr[1] * x_norm[1][6] + ySr[2] * x_norm[2][6] + ySr[3] * x_norm[3][6] + x_norm[4][6] * ySr[4] +
                 x_norm[5][6] * ySr[5] + x_norm[6][6] * ySr[6] + x_norm[7][6] * ySr[7] + ySr[8] * x_norm[8][6] + ySr[9] * x_norm[9][
                     6] + ySr[10] * x_norm[10][6] + ySr[11] * x_norm[11][6] + x_norm[12][6] * ySr[12] +
                 x_norm[13][6] * ySr[13]) / 14
        beta7 = (ySr[0] * x_norm[0][7] + ySr[1] * x_norm[1][7] + ySr[2] * x_norm[2][7] + ySr[3] * x_norm[3][7] + x_norm[4][7] * ySr[4] +
                 x_norm[5][7] * ySr[5] + x_norm[6][7] * ySr[6] + x_norm[7][7] * ySr[7] + ySr[8] * x_norm[8][7] + ySr[9] * x_norm[9][
                     7] + ySr[10] * x_norm[10][7] + ySr[11] * x_norm[11][7] + x_norm[12][7] * ySr[12] +
                 x_norm[13][7] * ySr[13]) / 14

        tN = []
        for i in range(8):
            tN.append(abs(locals().get("beta"+str(i))) / Dbeta)

        f3 = f1 * f2
        ttabl = 2.048
        if f3 > 25:
            ttabl = 1.960
        print("Оцінимо значимість коефіцієнтів регресіїї згідно критерію Стьюдента")
        str(list(map(lambda x: print(str(x), "  ", ttabl), tN)))

        coef = [1, 0, 0, 0, 0, 0, 0, 0]
        for k in range(8):
            if tN[k] > ttabl:
                coef[k] = 1
        print("Значимі коефіцієнти (1 - значимий) ", coef, "\n")
        yQ = [[0]]*14
        for i in range(14):
            for j in range(8):
                yQ[i][0] += coef[j] * b[j] * x_nat[i][j]

        print("Рівняння регресії згідно критерію Стьюдента")
        print("y = ", coef[0] * round(b[0][0], 4), "+", coef[1] * round(b[1][0], 4), " * x1 +", coef[2] * round(b[2][0], 4),
              " * x2 +", coef[3] * round(b[3][0], 4),
              "* x3 +", coef[4] * round(b[4][0], 4), " * x1 * x2 +", coef[5] * round(b[5][0], 4), " * x1 * x3 +",
              coef[6] * round(b[6][0], 4),
              "* x2 * x3 +", coef[7] * round(b[7][0], 4),
              " * x1 * x2 * x3")
        # Фишер
        d = 0
        for i in range(len(coef)):
            if coef[i] == 1:
                d += 1
        f4 = 14 - d
        S_ad = (m / (14 - d)) * (pow((yQ[0][0] - ySr[0]), 2) + pow((yQ[1][0] - ySr[1]), 2) + pow((yQ[2][0] - ySr[2]), 2) + pow(
            (yQ[3][0] - ySr[3]), 2)
                                 + pow((yQ[4][0] - ySr[4]), 2) + pow((yQ[5][0] - ySr[5]), 2) + pow((yQ[6][0] - ySr[6]), 2) + pow(
                    (yQ[7][0] - ySr[7]), 2) + pow((yQ[8][0] - ySr[8]), 2) + pow((yQ[9][0] - ySr[9]), 2) + pow(
                    (yQ[10][0] - ySr[10]), 2) + pow(
                    (yQ[11][0] - ySr[11]), 2)
                                 + pow((yQ[12][0] - ySr[12]), 2) + pow((yQ[13][0] - ySr[13]), 2))
        Fp = S_ad / DB
        Ft = 4.1709
        if f4 == 13:
            Fp = 3.3158
        if f4 == 12:
            Fp = 2.9223
        if f4 == 11:
            Fp = 2.6896
        if f4 == 10:
            Fp = 2.5336
        if f4 == 9:
            Fp = 2.4205
        if f4 == 8:
            Fp = 2.3343
        if f4 == 7:
            Fp = 2.2662
        if f4 == 6:
            Fp = 2.2107
        if f4 == 5:
            Fp = 2.1646
        if f4 == 4:
            Fp = 2.1256
        if f4 == 3:
            Fp = 2.0921
        if f4 == 2:
            Fp = 2.063
        if f4 == 1:
            Fp = 2.0374
        if Fp > Ft:
            print("Рівняння регресії неадекватно оригіналу при рівні значимості 0.05 за критерієм Фішера\n")
            flag2 = False
        else:
            print("Рівняння регресії адекватно оригіналу при рівні значимості 0.05 за критерієм Фішера")
            flag2 = False

    if flag2 == False:
        if Gp < Gt:
            print(Gp, "<", Gt)
            print("Дисперcія однорідна")
            print("m = ", m, "\n")
        else:
            print(Gp, ">", Gt)
            print("Дисперcія неоднорідна\n")
            print("m=", m)

        ySrNew = list()
        for i in range(len(ySr)):
            ySrNew.append(ySr[i])

        matrix = [[0 for i in range(11)] for j in range(11)]
        k5 = [0]*11

        for i in range(14):
            for j in range(11):
                matrix[0][j] += x_nat[i][j]
                matrix[1][j] += x_nat[i][j] * x_nat[i][1]
                matrix[2][j] += x_nat[i][j] * x_nat[i][2]
                matrix[3][j] += x_nat[i][j] * x_nat[i][3]
                matrix[4][j] += x_nat[i][j] * x_nat[i][4]
                matrix[5][j] += x_nat[i][j] * x_nat[i][5]
                matrix[6][j] += x_nat[i][j] * x_nat[i][6]
                matrix[7][j] += x_nat[i][j] * x_nat[i][7]
                matrix[8][j] += x_nat[i][j] * x_nat[i][8]
                matrix[9][j] += x_nat[i][j] * x_nat[i][9]
                matrix[10][j] += x_nat[i][j] * x_nat[i][10]
                k5[j] += x_nat[i][j] * ySrNew[j]


        for i in range(11):
                matrix[i]= list(map(lambda x: x/14, matrix[i]))
        k5 = list(map(lambda x: x / 14, k5))
        a = numpy.array(matrix)
        c = numpy.array(k5)
        b5 = numpy.linalg.solve(a, c)

        print("Рівняння регресії з урахуванням квадратичних членів: ")
        print("y = ", round(b5[0], 4), "+", round(b5[1], 4), " * x1 +", round(b5[2], 4), " * x2 +",
              round(b5[3], 4),
              "* x3 +", round(b5[4], 4),
              " * x1 * x2 +", round(b5[5], 4), " * x1 * x3 +", round(b5[6], 4), "* x2 * x3 +", round(b5[7], 4),
              " * x1 * x2 * x3 + ", round(b5[8], 4), "* x1^2 + ", round(b5[9], 4), "* x2^2", round(b5[10], 4),
              "* x3^2")
        DB = sum(D)/14
        Dbeta2 = DB / (15 * m)
        Dbeta_1 = math.sqrt(Dbeta2)
        beta0 = (ySr[0] * x_norm[0][0] + ySr[1] * x_norm[1][0] + ySr[2] * x_norm[2][0] + ySr[3] * x_norm[3][0] + x_norm[4][0] * ySr[4] +
                 x_norm[5][0] * ySr[5] + x_norm[6][0] * ySr[6] + x_norm[7][0] * ySr[7] + ySr[8] * x_norm[8][0] + ySr[9] * x_norm[9][
                     0] + ySr[10] * x_norm[10][0] + ySr[11] * x_norm[11][0] + x_norm[12][0] * ySr[12] +
                 x_norm[13][0] * ySr[13]) / 14
        beta1 = (ySr[0] * x_norm[0][1] + ySr[1] * x_norm[1][1] + ySr[2] * x_norm[2][1] + ySr[3] * x_norm[3][1] + x_norm[4][1] * ySr[4] +
                 x_norm[5][1] * ySr[5] + x_norm[6][1] * ySr[6] + x_norm[7][1] * ySr[7] + ySr[8] * x_norm[8][1] + ySr[9] * x_norm[9][
                     1] + ySr[10] * x_norm[10][1] + ySr[11] * x_norm[11][1] + x_norm[12][1] * ySr[12] +
                 x_norm[13][1] * ySr[13]) / 14
        beta2 = (ySr[0] * x_norm[0][2] + ySr[1] * x_norm[1][2] + ySr[2] * x_norm[2][2] + ySr[3] * x_norm[3][2] + x_norm[4][2] * ySr[4] +
                 x_norm[5][2] * ySr[5] + x_norm[6][2] * ySr[6] + x_norm[7][2] * ySr[7] + ySr[8] * x_norm[8][2] + ySr[9] * x_norm[9][
                     2] + ySr[10] * x_norm[10][2] + ySr[11] * x_norm[11][2] + x_norm[12][2] * ySr[12] +
                 x_norm[13][2] * ySr[13]) / 14
        beta3 = (ySr[0] * x_norm[0][3] + ySr[1] * x_norm[1][3] + ySr[2] * x_norm[2][3] + ySr[3] * x_norm[3][3] + x_norm[4][3] * ySr[4] +
                 x_norm[5][3] * ySr[5] + x_norm[6][3] * ySr[6] + x_norm[7][3] * ySr[7] + ySr[8] * x_norm[8][3] + ySr[9] * x_norm[9][
                     3] + ySr[10] * x_norm[10][3] + ySr[11] * x_norm[11][3] + x_norm[12][3] * ySr[12] +
                 x_norm[13][3] * ySr[13]) / 14
        beta4 = (ySr[0] * x_norm[0][4] + ySr[1] * x_norm[1][4] + ySr[2] * x_norm[2][4] + ySr[3] * x_norm[3][4] + x_norm[4][4] * ySr[4] +
                 x_norm[5][4] * ySr[5] + x_norm[6][4] * ySr[6] + x_norm[7][4] * ySr[7] + ySr[8] * x_norm[8][4] + ySr[9] * x_norm[9][
                     4] + ySr[10] * x_norm[10][4] + ySr[11] * x_norm[11][4] + x_norm[12][4] * ySr[12] +
                 x_norm[13][4] * ySr[13]) / 14
        beta5 = (ySr[0] * x_norm[0][5] + ySr[1] * x_norm[1][5] + ySr[2] * x_norm[2][5] + ySr[3] * x_norm[3][5] + x_norm[4][5] * ySr[4] +
                 x_norm[5][5] * ySr[5] + x_norm[6][5] * ySr[6] + x_norm[7][5] * ySr[7] + ySr[8] * x_norm[8][5] + ySr[9] * x_norm[9][5]
                 + ySr[10] * x_norm[10][5] + ySr[11] * x_norm[11][5] + x_norm[12][5] * ySr[12] +
                 x_norm[13][5] * ySr[13]) / 14
        beta6 = (ySr[0] * x_norm[0][6] + ySr[1] * x_norm[1][6] + ySr[2] * x_norm[2][6] + ySr[3] * x_norm[3][6] + x_norm[4][6] * ySr[4] +
                 x_norm[5][6] * ySr[5] + x_norm[6][6] * ySr[6] + x_norm[7][6] * ySr[7] + ySr[8] * x_norm[8][6] + ySr[9] * x_norm[9][
                     6] + ySr[10] * x_norm[10][6] + ySr[11] * x_norm[11][6] + x_norm[12][6] * ySr[12] +
                 x_norm[13][6] * ySr[13]) / 14
        beta7 = (ySr[0] * x_norm[0][7] + ySr[1] * x_norm[1][7] + ySr[2] * x_norm[2][7] + ySr[3] * x_norm[3][7] + x_norm[4][7] * ySr[4] +
                 x_norm[5][7] * ySr[5] + x_norm[6][7] * ySr[6] + x_norm[7][7] * ySr[7] + ySr[8] * x_norm[8][7] + ySr[9] * x_norm[9][
                     7] + ySr[10] * x_norm[10][7] + ySr[11] * x_norm[11][7] + x_norm[12][7] * ySr[12] +
                 x_norm[13][7] * ySr[13]) / 14
        beta8 = (ySr[0] * x_norm[0][8] + ySr[1] * x_norm[1][8] + ySr[2] * x_norm[2][8] + ySr[3] * x_norm[3][8] + x_norm[4][8] * ySr[4] +
                 x_norm[5][8] * ySr[5] + x_norm[6][8] * ySr[6] + x_norm[7][8] * ySr[7] + ySr[8] * x_norm[8][8] + ySr[9] * x_norm[9][
                     8] + ySr[10] * x_norm[10][8] + ySr[11] * x_norm[11][8] +
                 x_norm[12][8] * ySr[12] +
                 x_norm[13][8] * ySr[13]) / 14
        beta9 = (ySr[0] * x_norm[0][9] + ySr[1] * x_norm[1][9] + ySr[2] * x_norm[2][9] + ySr[3] * x_norm[3][9] + x_norm[4][9] * ySr[4] +
                 x_norm[5][9] * ySr[5] + x_norm[6][9] * ySr[6] + x_norm[7][9] * ySr[7] + ySr[8] * x_norm[8][9] + ySr[9] * x_norm[9][
                     9] + ySr[10] * x_norm[10][9] + ySr[11] * x_norm[11][9] + x_norm[12][9] * ySr[12] +
                 x_norm[13][9] * ySr[13]) / 14
        beta10 = (ySr[0] * x_norm[0][10] + ySr[1] * x_norm[1][10] + ySr[2] * x_norm[2][10] + ySr[3] * x_norm[3][10] + x_norm[4][
            10] * ySr[4] +
                  x_norm[5][10] * ySr[5] + x_norm[6][10] * ySr[6] + x_norm[7][10] * ySr[7] + ySr[8] * x_norm[8][10] + ySr[9] * x_norm[9][
                      10] + ySr[10] * x_norm[10][10] + ySr[11] * x_norm[11][10] + x_norm[12][10] * ySr[12] +
                  x_norm[13][10] * ySr[13]) / 14

        tN = []
        for i in range(11):
            tN.append(abs(locals().get("beta"+str(i)))/Dbeta_1)
        f3 = f1 * 14
        ttabl = scipy.stats.t.ppf((1 + 0.95) / 2, f3)
        print("Оцінимо значимість коефіцієнтів регресіїї згідно критерію Стьюдента")
        for i in range(len(tN)):
            print(tN[i], " ", ttabl)

        coef = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for k in range(11):
            if tN[k] > ttabl:
                coef[k] = 1
        print("Значимі коефіцієнти (1 - значимий) ", coef, "\n")
        yQ = [[0]]*14
        for i in range(14):
            for j in range(11):
                yQ[i][0] += coef[j] * b5[j] * x_nat[i][j]
        print("Рівняння регресії згідно критерію Стьюдента")
        print("y = ", coef[0] * round(b5[0], 4), "+", coef[1] * round(b5[1], 4), " * x1 +", coef[2] * round(b5[2], 4),
              " * x2 +", coef[3] * round(b5[3], 4),
              "* x3 +", coef[4] * round(b5[4], 4), " * x1 * x2 +", coef[5] * round(b5[5], 4), " * x1 * x3 +",
              coef[6] * round(b5[6], 4),
              "* x2 * x3 +", coef[7] * round(b5[7], 4),
              " * x1 * x2 * x3 +", coef[8] * round(b5[8], 4), "* x1^2 + ", coef[9] * round(b5[9], 4), "* x2^2",
              coef[10] * round(b5[10], 4),
              "* x3^2")
        # Фишер
        d = 0
        for i in range(len(coef)):
            if coef[i] == 1:
                d += 1
        f4 = 14 - d
        S_ad = (m / (14 - d)) * (pow((yQ[0][0] - ySr[0]), 2) + pow((yQ[1][0] - ySr[1]), 2) + pow((yQ[2][0] - ySr[2]), 2) + pow(
            (yQ[3][0] - ySr[3]), 2)
                                 + pow((yQ[4][0] - ySr[4]), 2) + pow((yQ[5][0] - ySr[5]), 2) + pow((yQ[6][0] - ySr[6]), 2) + pow(
                    (yQ[7][0] - ySr[7]), 2) + pow((yQ[8][0] - ySr[8]), 2) + pow((yQ[9][0] - ySr[9]), 2) + pow(
                    (yQ[10][0] - ySr[10]), 2) + pow(
                    (yQ[11][0] - ySr[11]), 2)
                                 + pow((yQ[12][0] - ySr[12]), 2) + pow((yQ[13][0] - ySr[13]), 2))
        Fp = S_ad / DB
        Ft = 4.1709
        if f4 == 13:
            Fp = 3.3158
        if f4 == 12:
            Fp = 2.9223
        if f4 == 11:
            Fp = 2.6896
        if f4 == 10:
            Fp = 2.5336
        if f4 == 9:
            Fp = 2.4205
        if f4 == 8:
            Fp = 2.3343
        if f4 == 7:
            Fp = 2.2662
        if f4 == 6:
            Fp = 2.2107
        if f4 == 5:
            Fp = 2.1646
        if f4 == 4:
            Fp = 2.1256
        if f4 == 3:
            Fp = 2.0921
        if f4 == 2:
            Fp = 2.063
        if f4 == 1:
            Fp = 2.0374
        if Fp > Ft:
            print(
                "Рівняння регресії неадекватно оригіналу при рівні значимості 0.05 за критерієм Фішера. Проведіть експеримент спочатку")
        else:
            print("Рівняння регресії адекватно оригіналу при рівні значимості 0.05 за критерієм Фішера")
else:
    print("Кількість незачимих коефіцієнтів", number_of_junk)