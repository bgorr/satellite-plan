import numpy as np
from math import factorial

sum = 0
for i in range(16):
    print(i)
    c = factorial(i+4)/(factorial(4)*factorial(i))
    sum += c

print(sum)

sum = 0
for i in range(7):
    print(i)
    c = factorial(i+2)/(factorial(2)*factorial(i))
    sum += c

print(sum)

n = 6
result = 0.5*factorial(n)/(factorial(int(n/2)))
print(result)
result2 = factorial(2*n)/(factorial(n)*factorial(n))
print(result2)

sum = 0
for i in range(7):
    c = factorial(i+2)/(factorial(2)*factorial(i))
    sum += c
print(sum)

sum = 0
for i in range(7):
    for j in range(i,16):
        sum += (factorial(j-i+1)/factorial(j-i) * factorial(i+2)/2/factorial(i))
print(sum)