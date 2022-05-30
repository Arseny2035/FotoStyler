import tensorflow as tf

import matplotlib.pyplot as plt

from imageio import mimsave
from IPython.display import display as display_fn
from IPython.display import Image, clear_output

import numpy as np
import os

from PIL import Image


def tensor_to_image(tensor):
    '''converts a tensor to an image'''
    tensor_shape = tf.shape(tensor)
    number_elem_shape = tf.shape(tensor_shape)
    if number_elem_shape > 3:
        assert tensor_shape[0] == 1
        tensor = tensor[0]
    return tf.keras.preprocessing.image.array_to_img(tensor)


def load_img(path_to_img):
    '''loads an image as a tensor and scales it to 512 pixels'''
    max_dim = 512
    image = tf.io.read_file(path_to_img)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.shape(image)[:-1]
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    image = tf.image.convert_image_dtype(image, tf.uint8)

    return image


def load_images(content_path, style_path):
    '''loads the content and path images as tensors'''
    content_image = load_img("{}".format(content_path))
    style_image = load_img("{}".format(style_path))

    return content_image, style_image


def imshow(image, title=None):
    '''displays an image with a corresponding title'''
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def show_images_with_objects(images, titles=[]):
    '''displays a row of images with corresponding titles'''
    if len(images) != len(titles):
        return

    plt.figure(figsize=(20, 12))
    for idx, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), idx + 1)
        plt.xticks([])
        plt.yticks([])
        imshow(image, title)
        plt.show()


def display_gif(gif_path):
    '''displays the generated images as an animated gif'''
    with open(gif_path, 'rb') as f:
        display_fn(Image(data=f.read(), format='png'))


def create_gif(gif_path, images):
    '''creates animation of generated images'''
    mimsave(gif_path, images, fps=1)

    return gif_path


def clip_image_values(image, min_value=0.0, max_value=255.0):
    '''clips the image pixel values by the given min and max'''
    return tf.clip_by_value(image, clip_value_min=min_value, clip_value_max=max_value)


def preprocess_image(image):
    '''centers the pixel values of a given image to use with VGG-19'''
    image = tf.cast(image, dtype=tf.float32)
    image = tf.keras.applications.vgg19.preprocess_input(image)

    return image


def get_style_loss(features, targets):
    """Expects two images of dimension h, w, c

    Args:
      features: tensor with shape: (height, width, channels)
      targets: tensor with shape: (height, width, channels)

    Returns:
      style loss (scalar)
    """
    # get the average of the squared errors
    style_loss = tf.reduce_mean(tf.square(features - targets))

    return style_loss


def get_content_loss(features, targets):
    """Expects two images of dimension h, w, c

    Args:
      features: tensor with shape: (height, width, channels)
      targets: tensor with shape: (height, width, channels)

    Returns:
      content loss (scalar)
    """
    # get the sum of the squared error multiplied by a scaling factor
    content_loss = 0.5 * tf.reduce_sum(tf.square(features - targets))

    return content_loss


def gram_matrix(input_tensor):
    """ Calculates the gram matrix and divides by the number of locations
    Args:
      input_tensor: tensor of shape (batch, height, width, channels)

    Returns:
      scaled_gram: gram matrix divided by the number of locations
    """

    # calculate the gram matrix of the input tensor
    gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

    # get the height and width of the input tensor
    input_shape = tf.shape(input_tensor)
    height = input_shape[1]
    width = input_shape[2]

    # get the number of locations (height times width), and cast it as a tf.float32
    num_locations = tf.cast(height * width, tf.float32)

    # scale the gram matrix by dividing by the number of locations
    scaled_gram = gram / num_locations

    return scaled_gram


def get_style_image_features(image, model, num_style_layers):
    """ Get the style image features

    Args:
      image: an input image

    Returns:
      gram_style_features: the style features as gram matrices
    """
    # preprocess the image using the given preprocessing function
    preprocessed_style_image = preprocess_image(image)

    # get the outputs from the custom vgg model that you created using vgg_model()
    outputs = model(preprocessed_style_image)

    # Get just the style feature layers (exclude the content layer)
    style_outputs = outputs[:num_style_layers]

    # for each style layer, calculate the gram matrix for that layer and store these results in a list
    gram_style_features = [gram_matrix(style_layer) for style_layer in style_outputs]

    return gram_style_features


def get_content_image_features(image, model, num_style_layers):
    """ Get the content image features

    Args:
      image: an input image

    Returns:
      content_outputs: the content features of the image
    """
    # preprocess the image
    preprocessed_content_image = preprocess_image(image)

    # get the outputs from the vgg model
    outputs = model(preprocessed_content_image)

    # get the content layers of the outputs
    content_outputs = outputs[num_style_layers:]

    # return the content layer outputs of the content image
    return content_outputs


def get_style_content_loss(style_targets, style_outputs, content_targets,
                           content_outputs, style_weight, content_weight,
                           num_style_layers, num_content_layers):
    """ Combine the style and content loss

    Args:
      style_targets: style features of the style image
      style_outputs: style features of the generated image
      content_targets: content features of the content image
      content_outputs: content features of the generated image
      style_weight: weight given to the style loss
      content_weight: weight given to the content loss

    Returns:
      total_loss: the combined style and content loss

    """

    # sum of the style losses
    style_loss = tf.add_n([get_style_loss(style_output, style_target)
                           for style_output, style_target in zip(style_outputs, style_targets)])

    # Sum up the content losses
    content_loss = tf.add_n([get_content_loss(content_output, content_target)
                             for content_output, content_target in zip(content_outputs, content_targets)])

    # scale the style loss by multiplying by the style weight and dividing by the number of style layers
    style_loss = style_loss * style_weight / num_style_layers

    # scale the content loss by multiplying by the content weight and dividing by the number of content layers
    content_loss = content_loss * content_weight / num_content_layers

    # sum up the style and content losses
    total_loss = style_loss + content_loss

    return total_loss


def update_image_with_style(image, style_targets, content_targets, style_weight,
                            var_weight, content_weight, optimizer, model, num_style_layers,
                            num_content_layers, with_variations):
    """
    Args:
      image: generated image
      style_targets: style features of the style image
      content_targets: content features of the content image
      style_weight: weight given to the style loss
      content_weight: weight given to the content loss
      var_weight: weight given to the total variation loss
      optimizer: optimizer for updating the input image
    """

    # calculate gradients using the function that you just defined.
    gradients = calculate_gradients(image, style_targets, content_targets,
                                    style_weight, content_weight, var_weight,
                                    model, num_style_layers, num_content_layers, with_variations)
    # print('gradients:', gradients)
    # print('image: ', image)

    # apply the gradients to the given image
    optimizer.apply_gradients([(gradients, image)])
    # print('image after optimizer: ', image)

    # clip the image using the utility clip_image_values() function
    image.assign(clip_image_values(image, min_value=0.0, max_value=255.0))
    # print('image after assign: ', image)

    return image


def fit_style_transfer(model, num_style_layers, num_content_layers, style_image, content_image, style_weight=1e-2,
                       content_weight=1e-4, var_weight=0, optimizer='adam', epochs=1, steps_per_epoch=1,
                       with_variations=False):
    """ Performs neural style transfer.
    Args:
      style_image: image to get style features from
      content_image: image to stylize
      style_targets: style features of the style image
      content_targets: content features of the content image
      style_weight: weight given to the style loss
      content_weight: weight given to the content loss
      var_weight: weight given to the total variation loss
      optimizer: optimizer for updating the input image
      epochs: number of epochs
      steps_per_epoch = steps per epoch

    Returns:
      generated_image: generated image at final epoch
      images: collection of generated images per epoch
    """

    images = []
    step = 0

    # get the style image features
    style_targets = get_style_image_features(style_image, model, num_style_layers)

    # get the content image features
    content_targets = get_content_image_features(content_image, model, num_style_layers)

    # initialize the generated image for updates
    generated_image = tf.cast(content_image, dtype=tf.float32)
    generated_image = tf.Variable(generated_image)

    # collect the image updates starting from the content image
    images.append(content_image)

    if with_variations:
        prefix = 'with_variations_'
    else:
        prefix = 'intermediate_'

    print("Start training")

    # incrementally update the content image with the style features
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1

            # Update the image with the style using the function that you defined
            generated_image = update_image_with_style(generated_image, style_targets, content_targets,
                                                      style_weight, var_weight, content_weight, optimizer,
                                                      model, num_style_layers, num_content_layers, with_variations)

            # print("generated_image: ", generated_image)

            ######################################
            if (m + 1) % 10 == 0:
                with Image.fromarray(np.squeeze(generated_image.numpy().astype(np.uint8), axis=0)) as img:
                    filename = prefix + str(n) + "_" + str(m) + ".jpg"
                    img.save(os.path.join("Temp_images", filename))
            #####################################

            print(".", end='')

            images.append(generated_image)

        # display the current stylized image
        # clear_output(wait=True)
        # display_image = tensor_to_image(generated_image)
        # display_fn(display_image)

        # append to the image collection for visualization later
        images.append(generated_image)
        print("Train step: {}".format(step))

    # convert to uint8 (expected dtype for images with pixels in the range [0,255])
    generated_image = tf.cast(generated_image, dtype=tf.uint8)

    return generated_image, images


# Plot Utilities
def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var


def plot_deltas_for_single_image(x_deltas, y_deltas, name="Original", row=1):
    plt.figure(figsize=(14, 10))
    plt.subplot(row, 2, 1)
    plt.yticks([])
    plt.xticks([])

    clipped_y_deltas = clip_image_values(2 * y_deltas + 0.5, min_value=0.0, max_value=1.0)
    imshow(clipped_y_deltas, "Horizontal Deltas: {}".format(name))

    plt.subplot(row, 2, 2)
    plt.yticks([])
    plt.xticks([])

    clipped_x_deltas = clip_image_values(2 * x_deltas + 0.5, min_value=0.0, max_value=1.0)
    imshow(clipped_x_deltas, "Vertical Deltas: {}".format(name))
    plt.show()


def plot_deltas(original_image_deltas, stylized_image_deltas):
    orig_x_deltas, orig_y_deltas = original_image_deltas

    stylized_x_deltas, stylized_y_deltas = stylized_image_deltas

    plot_deltas_for_single_image(orig_x_deltas, orig_y_deltas, name="Original")
    plot_deltas_for_single_image(stylized_x_deltas, stylized_y_deltas, name="Stylized Image", row=2)


def calculate_gradients(image, style_targets, content_targets,
                        style_weight, content_weight, var_weight,
                        model, num_style_layers, num_content_layers,
                        with_variations):
    """ Calculate the gradients of the loss with respect to the generated image
    Args:
      image: generated image
      style_targets: style features of the style image
      content_targets: content features of the content image
      style_weight: weight given to the style loss
      content_weight: weight given to the content loss
      var_weight: weight given to the total variation loss

    Returns:
      gradients: gradients of the loss with respect to the input image
    """
    with tf.GradientTape() as tape:
        # get the style image features
        style_features = get_style_image_features(image, model, num_style_layers)

        # get the content image features
        content_features = get_content_image_features(image, model, num_style_layers)

        # get the style and content loss
        loss = get_style_content_loss(style_targets, style_features, content_targets,
                                      content_features, style_weight, content_weight,
                                      num_style_layers, num_content_layers)

        # add the total variation loss
        if with_variations:
            loss += var_weight * tf.image.total_variation(image)

        # calculate gradients of loss with respect to the image
    gradients = tape.gradient(loss, image)

    return gradients


def save_images_collection(images, prefix, path):
    prefix = prefix + '_'
    for i, image in enumerate(images):
        # print('image:' + str(i), image)
        with Image.fromarray(np.squeeze(image.numpy().astype(np.uint8), axis=0)) as img:
            # with Image.fromarray(image) as img:
            filename = prefix + str(i) + '.jpg'
            img.save(os.path.join(path, filename))

# gif_images = [np.squeeze(image.numpy().astype(np.uint8), axis=0) for image in display_images]
