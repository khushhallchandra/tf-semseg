import tensorflow as tf

def UpSampling2D(x_, upscale=(2,2)):
    h, w, _ = x_.get_shape().as_list()[1:]
    h_, w_  = h*upscale[0], w*upscale[1]

    return tf.image.resize_nearest_neighbor(x_, size=(h_, w_))
