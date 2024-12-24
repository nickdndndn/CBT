
def retrieve_2d_tuple(value):
    if isinstance(value, tuple) and len(value) == 2:
        return value
    elif not isinstance(value, tuple):
        return(value,value)
    else:
        raise Exception(f'Not a correct value : {value}')