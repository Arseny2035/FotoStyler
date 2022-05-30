import os
import wget

OTHER_IMAGES_DIR = 'Images\Other'

# create directory
try:
    os.mkdir(OTHER_IMAGES_DIR)
except:
    pass


# download images to the directory that just created
def img_other_download(url, dir=OTHER_IMAGES_DIR, pic_name=None):
    if pic_name == None:
        pic_name = os.path.basename(url)
    try:
        wget.download(url, os.path.join(dir, pic_name))
    except:
        print('Error while loading:' + url)

# img_other_download('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/MLColabImages/cat1.jpg', pic_name='cat.jpg')
# img_other_download('https://cdn.pixabay.com/photo/2017/02/28/23/00/swan-2107052_1280.jpg', pic_name='swan.jpg')
# img_other_download('https://i.dawn.com/large/2019/10/5db6a03a4c7e3.jpg', pic_name='tnj.jpg')
# img_other_download('https://cdn.pixabay.com/photo/2015/09/22/12/21/rudolph-951494_1280.jpg', pic_name='rudolph.jpg')
# img_other_download('https://cdn.pixabay.com/photo/2015/10/13/02/59/animals-985500_1280.jpg', pic_name='dynamite.jpg')
# img_other_download('https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg', pic_name='painting.jpg')
