import time
arr=[i*1.1 for i in range(9999999)]
start=time.time()

y=[0 for _ in range(len(arr))]
mid=time.time()
print(mid-start)
for i in range(len(arr)):
    y[i]=arr[i]
end=time.time()

print(end-start)