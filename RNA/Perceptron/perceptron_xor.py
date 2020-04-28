from random import choice
from numpy import array, dot, random

unit_step = lambda x: 0 if x <= 0 else 1

training_data = [
    (array([0,0]), 0),
    (array([0,1]), 1),
    (array([1,0]), 1),
    (array([1,1]), 0),
]

test_data = [
    (array([0,0]), 0),
    (array([0,1]), 1),
    (array([1,0]), 1),
    (array([1,1]), 0),
]

w = random.rand(2)/2
ap = 0.2
stop = 0.001
it = 0
errors = 0

while errors > stop or it == 0:
    it = it + 1
    errors = 0
    print('---------- Iteration %d',it)
    for i in range(4):
        x, expected = choice(training_data)
        result = dot(w, x)
        error = expected - unit_step(result)
        errors = errors + error*error
        w += ap * error * x
        #print data
        print(x)
        print(w)
        print(error)
    print('errors = %f' %errors)

print('--------- Results')
for x, _ in test_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x, result, unit_step(result)))
