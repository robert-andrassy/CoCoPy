#!/usr/bin/env python3
import sys
import numpy as np
from decimal import Decimal

def parse_quad(s, return_float=True, endian='='):
    """parse a floating point number in binary128 format as specified by IEEE 754-2008
    Arguments:
        s:            a string containing the binary representation of the floating point number (length: 16)
        return_float: return result as a Python float (default) otherwise Decimal is used
        endian:       can be <, >, or = (default)
    """
    if not isinstance(s, (str, bytes)):
        raise TypeError("argument must be a string")
    if len(s) != 16:
        raise ValueError("length of input must be 16")

    if endian == '=':
        endian = '<' if np.little_endian else '>'

    if endian == '<':
        swap = slice(None,None,-1)
    elif endian == '>':
        swap = slice(None)
    else:
        raise ValueError("unknown endianness")

    s = np.fromstring(s, dtype='u1')
    bits = ''.join([np.binary_repr(i,width=8)[swap] for i in s])
    bits = bits[swap]

    sign = int(bits[0], 2)

    exponent = int(bits[1:16],2)

    significand = int(bits[16:],2)

    if exponent == 0x7fff:
        if significand == 0:
            ret = Decimal(float('inf') * (-1)**sign)
        else:
            ret = Decimal(float('nan'))
    elif exponent == 0:
        ret = Decimal((sign, (1,), 0)) * Decimal(2)**(-16382) * (0 + Decimal(2)**(1-113) * significand)
    else:
        ret = Decimal((sign, (1,), 0)) * Decimal(2)**(exponent-16383) * (1+Decimal(2)**(1-113) * significand)

    if return_float:
        return float(ret)
    else:
        return ret
