from SpaceCharge import *
import time

start = time.time()

X,XX = SPACE_CHARGE(1e3, 1e-6, 10, 1e-12)

end = time.time()
print(end - start)
