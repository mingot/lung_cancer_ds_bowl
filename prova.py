import multiprocessing
from time import sleep

def ff(num):
    print "Processing %d" % num
    if num==5:
	sleep(2)
    return num, num*num

pool =  multiprocessing.Pool(4)
nums = zip(*pool.map(ff, range(10)))
pool.close()
pool.join()

print "Finished"
print nums
