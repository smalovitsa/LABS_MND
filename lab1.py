import random as r
array = []
a0 = 1
a1 = 2
a2 = 3
a3 = 4
print("Коефіцієнти a0 = {}, a1 = {}, a2 = {}, a3 = {}".format(a0,a1,a2,a3))
for i in range(8):
    array.append([])
    for j in range(3):
        array[i].append(r.randint(0,20))
print("Отриманий масив факторів\n")
for k in range(len(array)):
    print(array[k])

resultY = []
x0 =[]
dx = []
for res in range(8):
    y = a0 + a1 * array[res][0] + a2 * array[res][1] + a3 * array[res][2]
    resultY.append(y)
for i in range(3):
    x_max = array[0][i]
    x_min = array[0][i]
    for j in range(1, 8):
        x_max = max(x_max, array[j][i])
        x_min = min(x_min, array[j][i])
    x0.append((x_max+x_min)/2)
    dx.append(x0[i]-x_min)
Xn = [[(array[j][i]-x0[i])/dx[i] for i in range(3)] for j in range(8)]

print("\nDX   ",dx)
print("\nX0    ",x0)
print("\nY   ",resultY)
print("\nXn масив \n")
for var in range(len(Xn)):
    print(Xn[var])
Yet = a0 + a1 * x0[0] + a2 * x0[1] + a3 * x0[2]
print("\nYэт = ", Yet)
print(str(Yet)[::-1])