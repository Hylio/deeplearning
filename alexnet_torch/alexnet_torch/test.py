a,b,c = input().split()

a = int(a)
b = int(b)
c = int(c)

if a>b:
    a,b = b,a
if b<c:
    print(a,b,c)
elif c<a:
    print(c,a,b)
else :
    print(a,c,b)