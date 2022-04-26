import math
print('Demo7:')
num = []
for i in range(2, 101):
    for j in range(2,  int(math.sqrt(i)) + 1):
        if i % j == 0:
            break
        else:
            num.append(i)
print(num)