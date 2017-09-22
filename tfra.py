import struct
import numpy as np
import tensorflow as tf
FLAG_BIG_ENDIAN = 0x01
MAGIC_NUMBER = 8746397786917265778
dtype_kind_to_enum = {'i': 1, 'u': 2, 'f': 3, 'c': 4}
dtype_enum_to_name = {0: 'user', 1: 'int', 2: 'uint', 3: 'float', 4: 'complex'}


def read_ra_header(f):
    filemagic = f.read(8)
    h = dict()
    h['flags'] = struct.unpack('<Q', f.read(8))[0]
    h['eltype'] = struct.unpack('<Q', f.read(8))[0]
    h['elbyte'] = struct.unpack('<Q', f.read(8))[0]
    h['size'] = struct.unpack('<Q', f.read(8))[0]
    h['ndim'] = struct.unpack('<Q', f.read(8))[0]
    h['shape'] = []
    for d in range(h['ndim']):
        h['shape'].append(struct.unpack('<Q', f.read(8))[0])

    h['shape'] = h['shape'][::-1]
    return h


def read_ra_tf(filename):
    with open(filename, 'rb') as f:
        h = read_ra_header(f)

    if h['eltype'] == 0:
        raise TypeError('Unable to convert user data.')
    
    shape = h['shape']
    size = h['size']
    ndim = h['ndim']
    dtype_name = dtype_enum_to_name[h['eltype']]
    dtype_bits = h['elbyte'] * 8
    offset = 48 + 8 * ndim
    dtype = '%s%d' % (dtype_name, dtype_bits)

    with tf.name_scope('Read_{}'.format(filename)):
        bytes = tf.substr(tf.read_file(filename), offset, size)
    
        if dtype_name == 'complex':
            real_dtype = 'float%d' % (dtype_bits // 2)
            real_tensor = tf.decode_raw(bytes, real_dtype)
            print(real_dtype, size)
            tensor = tf.bitcast(tf.reshape(real_tensor, [-1, 2]), dtype)
        else:
            tensor = tf.decode_raw(bytes, dtype)
        
        return tf.reshape(tensor, shape)

    
def write_ra(filename, data):
    flags = 0
    if data.dtype.str[0] == '>':
        flags |= FLAG_BIG_ENDIAN
    try:
        eltype = dtype_kind_to_enum[data.dtype.kind]
    except KeyError:
        eltype = 0
    elbyte = data.dtype.itemsize
    size = data.size * elbyte
    ndim = len(data.shape)
    shape = np.array(data.shape).astype('uint64')
    with open(filename, 'wb') as f:
        f.write(struct.pack('<Q', MAGIC_NUMBER))
        f.write(struct.pack('<Q', flags))
        f.write(struct.pack('<Q', eltype))
        f.write(struct.pack('<Q', elbyte))
        f.write(struct.pack('<Q', size))
        f.write(struct.pack('<Q', ndim))
        f.write(shape[::-1].tobytes())
        f.write(data.tobytes())
