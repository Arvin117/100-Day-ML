from math import sqrt


def Demo1():
    """
        素数：只能被1和自身整除的大于1的证书
    """
    num = int(input('Input: '))
    end = int(sqrt(num))
    is_prime = True

    for x in range(2, end + 1):
        if num % x == 0:
            is_prime = False
            break
        if is_prime and num != 1:
            print(f'Yes! {num:d} is a prime number.')
        else:
            print(f'No! {num:d} is not a prime number')


def Demo2():
    """
    最大公约数和最小公倍数
    :return:
    """

    num1 = int(input('No.1:'))
    num2 = int(input('No.2'))
    if num1 > num2:
        num1, num2 = num2, num1

    for factor in range(num1, 0, -1):
        if num1 % factor == 0 and num2 % factor == 0:
            print(f'{num1:d}和{num2:d}的最大公约数是{factor:d}')
            print('%d和%d的最小公倍数是%d' % (num1, num2, num1 * num2 // factor))
            break

def Demo3():
    row = int(input('Input the row:'))
    for i in range(row):
        for _ in range(i + 1):
            print('*', end='')
        print()

if __name__ == '__main__':
    # Demo1()
    # Demo2()
    Demo3()