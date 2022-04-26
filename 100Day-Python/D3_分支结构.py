"""
        3x - 5  (x > 1)
f(x) =  x + 2   (-1 <= x <= 1)
        5x + 3  (x < -1)

x = float(input('x = '))
if x > 1:
    y = 3 * x - 5
elif x >= -1:
    y = x + 2
else:
    y = 5 * x + 3
print('f(%.2f) = %.2f' % (x, y))
"""
print("Demo 1 :英寸和厘米转换")
# value = float(input('请输入长度: '))
# unit = input('请输入单位: ')
value = 11
unit = 'cm'
if unit == 'in' or unit == '英寸':
    print('%.2f英寸 = %.2f厘米' % (value, value * 2.54))
elif unit == 'cm' or unit == '厘米':
    print('%.2f厘米 = %.2f英寸' % (value, value / 2.54))
else:
    print('请出入有效的单位（英寸或者厘米）')
print()

print('Demo 2 : 判断三边是否可以构成三角形')
# a, b, c = float(input('a = ')), float(input('b = ')), float(input('c = '))
a,b,c = (input("请输入三角形三边的长：").split())
# print(a, b, c)
# print(type(a))
a= int(a)
b= int(b)
c= int(c)
if a + b > c and a + c > b and b + c > a:
    print('周长 : %.2f' % (a + b + c))
    p = (a + b + c) / 2
    area = (p * (p - a) * (p - b) * (p - c)) ** 0.5
    print('面积: %.2f' % area)
else:
    print('No！')










