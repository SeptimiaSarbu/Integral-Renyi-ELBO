#Septimia Sarbu
#This code is built on the example taken from https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from scipy.special import erf
from scipy.special import dawsn

#This script computes the gradients of the error and Dawson functions, as RegisterGradient in tensorflow

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def numpy_erf(x):
    return erf(x)

def numpy_dawsn(x):
    return dawsn(x)

# Def custom function:
def myerf(x, name=None):

    with ops.name_scope(name, "Myerf", [x]) as name:
        erf_x = py_func(numpy_erf,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=_MyErfGrad)  # <-- here's the call to the gradient
        return erf_x[0]


# Actual gradient:
def _MyErfGrad(op, grad):
    x = op.inputs[0]
    grad_x = 2/np.sqrt(np.pi) * tf.exp(-(x**2))
    return grad * grad_x  # add your custom gradient here to use in the backpropagation/chain rule, along with the other gradients given by grad

# Def custom function:
def mydawson(x, name=None):

    with ops.name_scope(name, "Myerf", [x]) as name:
        dawson_x = py_func(numpy_dawsn,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=_MyDawsonGrad)  # <-- here's the call to the gradient
        return dawson_x[0]


# Actual gradient:
def _MyDawsonGrad(op, grad):
    x = op.inputs[0]
    grad_x = 1 - 2 * tf.multiply(x,tf.py_func(numpy_dawsn, [x], tf.float32))

    return grad * grad_x  # add your custom gradient here to use in the backpropagation/chain rule, along with the other gradients given by grad

with tf.Session() as sess:
    x = tf.constant([1., 2.])
    y = myerf(x)
    z = mydawson(x)
    tf.global_variables_initializer().run()

    #print(x.eval(), y.eval(), y_new.eval(), tf.gradients(y, x)[0].eval(), tf.gradients(y_new, x)[0].eval())
    print("\n x.eval()=", sess.run(x))
    print("\n y.eval()=", y.eval())
    print("\n tf.gradients(y, x)[0].eval()=", tf.gradients(y, x)[0].eval())
    print("\n z.eval()=", z.eval())
    print("\n tf.gradients(z, x)[0].eval()=", tf.gradients(z, x)[0].eval())
