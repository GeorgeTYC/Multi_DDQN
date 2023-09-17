from Test.test1 import test1
from Test.test2 import test2

test2 = test2()
test1 = test1()

testTemp = test2.cal(test1)

print(test1.get_order())
print(testTemp.get_order())