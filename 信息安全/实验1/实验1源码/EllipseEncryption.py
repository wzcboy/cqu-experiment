"""
实现椭圆加密算法
y^2 = x^3 + ax + b
"""
import random
import sys
# 全局参数
a = 7        # 椭圆参数
b = 13       # 椭圆参数
p = 137       # 素数
Gx = 5       # 基点横坐标
Gy = 6       # 基点纵坐标
k = 5        # 私钥

def inverse(n):
    """
    求一个数负元：n^-1，n * n^-1 = 1 (mod p)
    负元属于[0,p-1]
    :param n: 一个数字
    :param p: 素数p
    :return: 负元
    """
    for i in range(1, p):
        if (i * n) % p == 1:
            return i
    return -1


def gcd(a, b):
    """
    求a和b的最大公约数，辗转相除法
    :param a: 整数
    :param b: 整数
    :return: 最大公约数
    """
    while b != 0:
        temp = a % b
        a = b
        b = temp
    return a


def add(x1, y1, x2, y2):
    """
    将两个椭圆上的坐标相加
    P + Q = R, P(x1, y1), Q(x2, y2)
    :return: R的坐标(x3, y3)
    """
    # 先求P，Q连线的斜率K
    # 斜率的符号位,1代表正，-1代表负
    flag = 1

    # 当P==Q, k=(3(x1)^2 + a) / 2y1 mod p
    if x1 == x2 and y1 == y2:
        numerator = 3 * (x1 ** 2) + a
        denominator = 2 * y1

    # 当p!=Q，k=(y2-y1)/(x2-x1) mod p
    else :
        numerator = y2 - y1
        denominator = x2 - x1
        if numerator * denominator < 0:
            flag = -1
            numerator = abs(numerator)
            denominator = abs(denominator)

    gcdVal = gcd(numerator, denominator)
    numerator = numerator // gcdVal
    denominator = denominator // gcdVal

    # 分子/分母 = 分子* 分母^-1
    inverseDenominator = inverse(denominator)
    k = (flag * numerator * inverseDenominator) % p

    # 根据公式求R
    x3 = (k**2 - x1 - x2) % p
    y3 = (k * (x1 - x3) - y1) % p
    return x3, y3


def k_mul_G(k, Gx, Gy):
    """
    计算kG，G为椭圆上的坐标
    方法：循环计算G+G，直到k个加完
    :param k: 一个常数
    :param Gx: 横坐标
    :param Gy: 纵坐标
    :return: 坐标
    """
    sumX = Gx
    sumY = Gy
    while k > 1:
        sumX, sumY = add(sumX, sumY, Gx, Gy)
        k -= 1
    return sumX, sumY


def getRank(x0, y0):
    """
    求椭圆上坐标点(x0, y0)的阶
    即 nP = 0,方法是不断加P，直到等于-P
    :param x0: 横坐标
    :param y0: 纵坐标
    :return: 阶数
    """
    # -P的坐标
    x1 = x0
    y1 = (-y0) % p
    sumX = x0
    sumY = y0

    # 阶数
    n = 1
    while True:
        n += 1
        # P + P
        sumX, sumY = add(sumX, sumY, x0, y0)
        # 如果等于-P，则返回n+1
        if sumX == x1 and sumY == y1:
            return n + 1


def getAbelGroup():
    """
    得到椭圆上所有可取值的坐标
    :return:
    """
    group = []
    for i in range(p):
        for j in range(p):
            if (j**2) % p == (i**3 + a * i + b) % p:
                group.append((i, j))
    print(group)
    return group


def validate(group):
    """
    验证参数选择是否正确
    :param group: 所有椭圆上的坐标点
    :return: True or False
    """
    # 4a^3 + 26b^2 != 0
    val = (4 * (a**3) + 27 * (b**2)) % 19
    if val == 0:
        return False
    # 选取的基点G要求是椭圆上的点
    if (Gx, Gy) not in group:
        return False
    return True


if __name__ == '__main__':
    print("选取的参数：")
    print("a={} b={} p={} G=({}, {}), k={}\n".format(a, b, p, Gx, Gy, k))

    print("所有的可取的坐标点：")
    group = getAbelGroup()

    # 验证参数是否正确
    if not validate(group):
        print("选取的参数有误！")
        sys.exit()
    x3, y3 = add(3, 46, 4, 67)
    print(x3, y3)
    # 确定椭圆在点G的阶
    n = getRank(Gx, Gy)
    # 生成公开密钥 Q = kG
    Qx, Qy = k_mul_G(k, Gx, Gy)
    # 用户B将Ep(a, b), Q, G传给用户A

    # 用户A产生一个随机数r
    r = random.randint(1, n-1)
    # 用户A计算C1, C2,并将C1，C2传给用户B
    C1X, C1Y = k_mul_G(r, Qx, Qy)
    C2X, C2Y = k_mul_G(r, Gx, Gy)

    # 用户A进行明文加密
    while True:
        print("---------------------------------")
        plain_text = input("【用户A】请输入需要加密的字符串:").strip()
        cipher_text = []
        # 对每一个字符进行加密，对每一个字符转换成unicode编码，再乘上C1X和C1Y
        for char in plain_text:
            # 转换成unicode编码
            intchar = ord(char)
            cipher_char = intchar * C1X * C1Y
            cipher_text.append(cipher_char)

        # 打印密文
        # 模拟将密文和C2传给用户B
        print("【用户B】收到的密文为：", end='')
        for c in cipher_text:
            print(c, end='')
        print()
        print("【用户B】收到的C2为：({}, {})".format(C2X, C2Y))

        # 用户B对收到的密文进行解密
        # 根据私钥k和C2来计算
        print("【用户B】解密得到的明文：", end='')
        for char in cipher_text:
            decryptQX, decryptQY = k_mul_G(k, C2X, C2Y)
            decryptText = chr((char // decryptQY) // decryptQX)
            print(decryptText, end='')
        print("\n---------------------------------")