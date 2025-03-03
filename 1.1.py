#1.1
import random

def r():
    return random.randint(0, 100)

numbers = [r(), r(), r(), r(), r()]
print(f"生成的随机数列表: {numbers}")
sum_numbers = sum(numbers)
average_numbers = sum_numbers / len(numbers)
print(f"总和: {sum_numbers}")
print(f"平均值: {average_numbers:.2f}")

#1.2
# 方法1：使用步长参数
print("方法1：使用步长参数")
for i in range(2, 11, 2):  # 从2开始，步长为2，到11结束（不包含）
    print(i)

# 方法2：使用条件判断
print("方法2：使用条件判断")
for i in range(1, 11):
    if i % 2 == 0:  # 如果i除以2的余数为0，则是偶数
        print(i)

#1.3
# 正确的函数定义
def temperature_converter(C):
    return C * 9/5 + 32

# 创建函数别名（如果需要）
temp_conv = temperature_converter

# 测试函数
celsius = 25
fahrenheit = temperature_converter(celsius)
print(f"{celsius}°C 等于 {fahrenheit}°F")

# 使用别名测试
celsius = 30
fahrenheit = temp_conv(celsius)
print(f"{celsius}°C 等于 {fahrenheit}°F")
    