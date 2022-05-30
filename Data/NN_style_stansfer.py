import tensorflow as tf

from keras import backend as K
import numpy as np

import os

from Utilits import load_images, show_images_with_objects
from Utilits import fit_style_transfer, create_gif, display_gif
from Utilits import high_pass_x_y, plot_deltas, save_images_collection

from Data.Models import vgg_model

from Image_loader import OTHER_IMAGES_DIR

from PIL import Image

# set default images
content_path = os.path.join(OTHER_IMAGES_DIR, 'Ars.jpg')
style_path = os.path.join(OTHER_IMAGES_DIR, 'painting.jpg')

# display the content and style image
content_image, style_image = load_images(content_path, style_path)
show_images_with_objects([content_image, style_image],
                         titles=[f'content image: {content_path}',
                                 f'style image: {style_path}'])

# style layers of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

# choose the content layer and put in a list
content_layers = ['block5_conv2']

# combine the two lists (put the style layers before the content layers)
output_layers = style_layers + content_layers

K.clear_session()

# declare auxiliary variables holding the number of style and content layers
NUM_CONTENT_LAYERS = len(content_layers)
NUM_STYLE_LAYERS = len(style_layers)

IMG_PATH = 'Temp_images'

K.clear_session()

# create a vgg-19 model
vgg = vgg_model(output_layers)
vgg.summary()

tmp_layer_list = [layer.output for layer in vgg.layers]

# define style and content weight
style_weight = 2e-2
content_weight = 1e-2

# define optimizer. learning rate decreases per epoch.
adam = tf.optimizers.Adam(
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=20.0, decay_steps=100, decay_rate=0.50
    )
)

# start the neural style transfer
print('Start training network (intermediate...)')
stylized_image, display_images = fit_style_transfer(model=vgg, num_style_layers=NUM_STYLE_LAYERS,
                                                    num_content_layers=NUM_CONTENT_LAYERS,
                                                    style_image=style_image, content_image=content_image,
                                                    style_weight=style_weight, content_weight=content_weight,
                                                    var_weight=0, optimizer=adam,
                                                    epochs=20, steps_per_epoch=100)

# display GIF of Intermedite Outputs
# print('Saving gifs...')
# GIF_PATH = 'style_transfer.gif'
# gif_images = [np.squeeze(image.numpy().astype(np.uint8), axis=0) for image in display_images]
# gif_path = create_gif(GIF_PATH, gif_images)
# display_gif(gif_path)

# print('Saving images...')
# save_images_collection(display_images, 'Intermediate', IMG_PATH)

# Display the frequency variations

original_x_deltas, original_y_deltas = high_pass_x_y(
    tf.image.convert_image_dtype(content_image, dtype=tf.float32))

stylized_image_x_deltas, stylized_image_y_deltas = high_pass_x_y(
    tf.image.convert_image_dtype(stylized_image, dtype=tf.float32))

plot_deltas((original_x_deltas, original_y_deltas), (stylized_image_x_deltas, stylized_image_y_deltas))

style_weight = 2e-2
content_weight = 1e-2
var_weight = 2

adam = tf.optimizers.Adam(
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=20.0, decay_steps=100, decay_rate=0.50
    )
)
print('Start training network (regular)')
stylized_image_reg, display_images_reg = fit_style_transfer(model=vgg, num_style_layers=NUM_STYLE_LAYERS,
                                                            num_content_layers=NUM_CONTENT_LAYERS,
                                                            style_image=style_image, content_image=content_image,
                                                            style_weight=style_weight, content_weight=content_weight,
                                                            var_weight=var_weight, optimizer=adam,
                                                            epochs=40, steps_per_epoch=50,
                                                            with_variations=True)

# # Display GIF
# print('Saving gifs...')
# GIF_PATH = 'style_transfer_reg.gif'
#
# gif_images_reg = [np.squeeze(image.numpy().astype(np.uint8), axis=0) for image in display_images_reg]
# gif_path_reg = create_gif(GIF_PATH, gif_images_reg)
# display_gif(gif_path_reg)


print('Saving images...')
# save_images_collection(display_images_reg, 'Regular', IMG_PATH)


# Display Frequency Variations

original_x_deltas, original_y_deltas = high_pass_x_y(
    tf.image.convert_image_dtype(content_image, dtype=tf.float32))

stylized_image_reg_x_deltas, stylized_image_reg_y_deltas = high_pass_x_y(
    tf.image.convert_image_dtype(stylized_image_reg, dtype=tf.float32))

try:
    plot_deltas((original_x_deltas, original_y_deltas), (stylized_image_reg_x_deltas, stylized_image_reg_y_deltas))
except:
    print('Plot deltas error')

K.clear_session()

print('END')
