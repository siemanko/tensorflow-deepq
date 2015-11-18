import tensorflow as tf

def base_name(var):
    """Extracts value passed to name= when creating a variable"""
    return var.name.split('/')[-1].split(':')[0]

def copy_variables(variables):
    res = {}
    for v in variables:
        name = base_name(v)
        copied_var = tf.Variable(v.initialized_value(), name=name)
        res[name] = copied_var
    return res
