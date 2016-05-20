""" Theano implementation of pass quality """

import theano
from theano import tensor as tt
from theano import scan


def one_step(s, s_abs, tt_error, tt_r, tt_height, tt_current_error, tt_flag):
    """
    One step function using by pq_theano scan
    :param s:
    :param s_abs:
    :param tt_error:
    :param tt_r:
    :param tt_height:
    :param tt_current_error:
    :param tt_flag:
    :return:
    """

    tt_current_error = tt.switch(tt.isclose(tt_height, 2.) and tt.isclose(tt_flag, 0.), 2., tt_current_error)
    tt_flag += tt.switch(tt_height > 1., 1., 0.)

    tt_height = tt.switch(tt.ge(tt_height + s, 0.), tt_height + s, 0.)

    tt_current_error += s_abs

    tt_error += tt_current_error * tt.switch(tt.isclose(tt_current_error, 4.), 0., 1.) / 3. * tt.isclose(tt_height, 0.)
    tt_r += tt.isclose(tt_current_error, 4.) * tt.isclose(tt_height, 0.)
    tt_current_error -= tt_current_error * tt.isclose(tt_height, 0.)
    return tt_error, tt_r, tt_height, tt_current_error, tt_flag


def pq_theano(y_true, y_pred):
    """
    Theano implementation of Pass Quality function.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_pred = tt.ge(y_pred, tt.mean(y_pred)).T[-1].T
    y_true = y_true.T[-1].T

    tt_diffs = tt.extra_ops.diff(y_true + y_pred)

    tt_r = theano.shared(0., 'r')
    tt_height = theano.shared(0., 'h')
    tt_error = theano.shared(0., 'err')
    tt_current_error = theano.shared(0., 'c_err')
    tt_flag = theano.shared(0., 'flag')

    values, updates = scan(fn=one_step,
                           sequences=[tt_diffs, tt.abs_(tt_diffs)],
                           outputs_info=[tt_error,
                                         tt_r,
                                         tt_height,
                                         tt_current_error,
                                         tt_flag])

    epsilon = 0.0000000001

    tt_ret = (1 - (values[1][-1] + epsilon) / (values[1][-1] +
                                               values[0][-1] +
                                               epsilon))
    return tt_ret


# for theano.function compilation
def pq_theano_f(y_true, y_pred):

    y_pred = tt.ge(y_pred, tt.mean(y_pred))
    y_true = y_true

    tt_diffs = tt.extra_ops.diff(y_true + y_pred)

    # tt_r = tt.shape_padleft(theano.shared(0., 'r'))
    tt_r = theano.shared(0., 'r')
    # tt_height = tt.shape_padleft(theano.shared(0., 'h'))
    tt_height = theano.shared(0., 'h')
    # tt_error = tt.shape_padleft(theano.shared(0., 'err',))
    tt_error = theano.shared(0., 'err')
    # tt_current_error = tt.shape_padleft(theano.shared(0., 'c_err'))
    tt_current_error = theano.shared(0., 'c_err')
    # tt_ret = theano.tensor.col('ret')
    tt_flag = theano.shared(0., 'flag')

    values, updates = scan(fn=one_step,
                           sequences=[tt_diffs, tt.abs_(tt_diffs)],
                           outputs_info=[tt_error,
                                         tt_r,
                                         tt_height,
                                         tt_current_error,
                                         tt_flag])

    # print values[0].type

    epsilon = 0.0000000001
    # print tt.ones_like(values[0]).type
    # print values[1].type
    tt_ret = 1 - (values[1] + epsilon) / (values[1] +
                                          values[0] +
                                          epsilon)
    return tt_ret
