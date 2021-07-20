from dask.distributed import Client
from dask import delayed
from time import sleep

def inc(x):
    # sleep(1)
    return x + 1

def add(x, y):
    # sleep(1)
    return x + y

if __name__ == '__main__':
    client = Client(n_workers=4)
    x = delayed(inc)(1)
    y = delayed(inc)(2)
    print(x)
    print(y)
    z = delayed(add)(x, y)
    print(z)
    # print(z.compute())
    im = z.visualize()
    open('mydask.png', 'wb').write(im.data)
    client.close()
    