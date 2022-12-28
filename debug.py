
import inspect

import torch

def this_line():
    frame_info = inspect.stack()[1]
    return f'{frame_info.filename}:{frame_info.lineno}'

def caller_line():
    frame_info = inspect.stack()[2]
    return f'{frame_info.filename}:{frame_info.lineno}'

def show_msg(msg):
    str_caller_line = caller_line()
    print(f'\n>>> DEBUG >>> {str_caller_line}: \n{msg}')

def show_obj(**kwargs):
    str_caller_line = caller_line()
    for key, value in kwargs.items():
        print(f'\n>>> DEBUG >>> {str_caller_line}: \n{key}: {value}')

def show_sum(**kwargs):
    str_caller_line = caller_line()
    
    # Get the sum of all the inputs.
    print(f'\n>>> DEBUG >>> {str_caller_line}: sum of objects: ')
    for key, value in kwargs.items():
        print(f'{key}: {torch.sum(value)}')