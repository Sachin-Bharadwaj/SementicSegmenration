MOD_ID = 'id'
MOD_RGB = 'rgb'
MOD_SEMSEG = 'semseg'
MOD_DEPTH = 'depth'

INVALID_VALUE = float('-inf')

INTERP = {
    MOD_ID: None,
    MOD_RGB: 'bilinear',
    MOD_SEMSEG: 'nearest',
    MOD_DEPTH: 'nearest',
}