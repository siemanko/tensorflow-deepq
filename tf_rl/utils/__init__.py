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

def unpack_tf_givens(d):
    new_dict = {}
    for k,v in d.items():
        if isinstance(k,(tuple, list)):
            assert isinstance(v,(tuple,list)) and len(k) == len(v)
            for k_inner, v_inner in zip(k,v):
                new_dict[k_inner] = v_inner
        else:
            new_dict[k] = v
    return new_dict
