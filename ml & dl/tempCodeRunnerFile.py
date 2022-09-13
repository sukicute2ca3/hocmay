a = []
b = []
for i in range(int(input())):
    a.append(int(input()))
k = int(input())
if k >= len(a):
    for i in range(len(a)-1, -1, -1):
        print(a[i], end=" ")
elif k < len(a):
    for i in range(k):
        b.append(a[i])
    a = a + b
    for i in range(k, len(a)):
        print(a[i], end=" ")
