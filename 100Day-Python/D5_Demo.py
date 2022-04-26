import math
from random import randint


# 寻找水仙花数，三位数，每位数立方的和等于本身：153--->1^3 + 5^3 +3^3 = 153
def Demo1():
    print('Demo1: ')
    for num in range(100, 1000):
        low = num % 10  # 个位
        mid = num // 100 % 10  # 十位
        high = num // 100  # 百位
        if num == low ** 3 + mid ** 3 + high ** 3:
            print(num)
    print()

# 反转正整数
def Demo2():
    print('Demo2: ')
    # num = int(input('num = :'))
    num = 1234
    reversed_num = 0
    while num > 0:
        reversed_num = reversed_num * 10 + num % 10
        num //= 10
    print(num)
    print(reversed_num)
    print()


# 百钱百🐔，公鸡5块，母鸡3块，小鸡一块钱仨，100块钱买100只鸡，各几只
def Demo3():
    print('Demo3: ')
    for i in range(20):
        for j in range(33):
            z = 100 - 5 * i - 3 * j
            if z * 3 + j + i == 100:
                print('公鸡： %d, 母鸡： %d, 小鸡： %d' % (i, j, z * 3))
    print()


# 花旗骰 CRAPS：玩家第一次摇骰子如果摇出了7点或11点，玩家胜；玩家第一次如果摇出2点、3点或12点，庄家胜；
# 其他点数玩家继续摇骰子，如果玩家摇出了7点，庄家胜；如果玩家摇出了第一次摇的点数，玩家胜；
# 其他点数，玩家继续要骰子，直到分出胜负。
# 玩家开始有1000块钱，输光结束游戏
def Demo4():
    print('Demo4: ')
    money = 1000
    while money > 0:
        print('你还', money, '块')
        again = False
        while True:
            debt = int(input('下注：'))
            if 0 < debt <= 1000:
                break
        first = randint(1, 6) + randint(1, 6)
        print('点数：', first)
        if first == 7 or first == 11:
            print('You Win!')
            money += debt
        elif first == 2 or first == 3 or first == 12:
            print('Sorry!')
            money -= debt
        else:
            again = True
        while again:
            again = False
            current = randint(1, 6) + randint(1, 6)
            if current == 7:
                print('Sorry!')
                money -= debt
            elif current == first:
                print('You Win!')
                money += debt
            else:
                again = True
    print('破产了，别玩了！')
    print()


# 斐波那契数列
def Demo5():
    print('Demo5: ')
    result = []
    x1 = 1
    x2 = 1
    result.append(x1)
    result.append(x2)
    for i in range(1, 19):
        result.append(result[i] + result[i - 1])
    print('斐波那契数列：', result)
    print()


# 完美数：完美数又称为完全数或完备数，它的所有的真因子（即除了自身以外的因子）的和（即因子函数）恰好等于它本身。
# 例如：6（$6=1+2+3$）和28（$28=1+2+4+7+14$）就是完美数。完美数有很多神奇的特性，有兴趣的可以自行了解。
def Demo6():
    print('Demo6: ')
    perfect_num = {}
    for i in range(2, 10001):
        temp_num = []
        for j in range(1, int(i / 2) + 1):
            if i % j == 0:
                temp_num.append(j)

        if sum(temp_num) == i:
            print(f'{i} : {temp_num}')
            perfect_num[i] = temp_num
    print(perfect_num)
    print()


# 素数:素数指的是只能被1和自身整除的正整数（不包括1）。
def Demo7():
    print('Demo7:')
    count = 0
    num = []
    for x in range(2, 101):
        for i in range(2, int(pow(x, 0.5)) + 1):
            if x % i == 0:
                break
        else:
            count += 1
            num.append(x)

    print(count)
    print(num)
    print(list(x for x in range(2, 101) if 0 not in (x % i for i in range(2, x))))


if __name__ == '__main__':
    Demo1()
    Demo2()
    Demo3()
    # Demo4()
    Demo5()
    Demo6()
    Demo7()
