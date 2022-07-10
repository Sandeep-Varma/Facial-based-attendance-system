import multiprocessing
from functools import partial

def product(x,y):
	return x * y

if __name__ == '__main__':
	pool = multiprocessing.Pool()
	# pool = multiprocessing.Pool(processes=4)
	input1 = 100
	input2 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,10000,34789597,346234,5641]
	outputs_async = pool.map_async(partial(product,input1), input2)
	outputs = outputs_async.get()
	print(outputs)
