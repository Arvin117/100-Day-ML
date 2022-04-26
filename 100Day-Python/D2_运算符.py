import numpy as np

print("Demo 1 : 温度转换")
# f = float(input('请输入华氏温度：'))
f = 111
c = (f - 32) / 1.8
print('%.1f华氏度 = %.1f摄氏度' % (f, c))
print('{:.1f}华氏度 = {:.1f}摄氏度'.format(f, c))
print(f'{f:.1f}华氏度 = {c:.1f}摄氏度')
print()

print('Demo 2 : 圆的周长和面积')
# radius = float(input('请输入圆的半径：'))
radius = 3
perimeter = 2 * np.pi * radius
area = np.pi * radius**2
print('周长: %.2f' % perimeter)
print('面积: %.2f' % area)
print()

print('Demo 3 : 判断是不是闰年')
year = int(input('请输入年份： '))
is_leap = year % 4 == 0 and year % 100 != 0 or\
    year % 400 == 0
print(is_leap)