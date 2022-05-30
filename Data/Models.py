import tensorflow as tf

def vgg_model(layer_names):
    """ Creates a vgg model that outputs the style and content layer activations.

    Args:
      layer_names: a list of strings, representing the names of the desired content and style layers

    Returns:
      A model that takes the regular vgg19 input and outputs just the content and style layers.

    """

    # load the the pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')

    # freeze the weights of the model's layers (make them not trainable)
    vgg.trainable = False

    # create a list of layer objects that are specified by layer_names
    outputs = [vgg.get_layer(name).output for name in layer_names]

    # create the model that outputs content and style layers only
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    return model