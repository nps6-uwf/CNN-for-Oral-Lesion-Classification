import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def augment(dirName, fname, save_to_dir, count = 20):
    if not os.path.exists(save_to_dir): # make save_to_directory if it doesn't exist
        os.mkdir(save_to_dir)
    for imgName in os.listdir(dirName):
        img = load_img(os.path.join(dirName, imgName))  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir=save_to_dir, save_prefix=fname, save_format='png'):
            i += 1
            if i > count:
                break  # otherwise the generator would loop indefinitely

def main():
    
    #dirName = "../data/traintest/traintest_lesion_v_nonlesion/lesion/"
    #fname = "herpes"
    #save_to_dir = "augmented_herpes"
    dirName = "../data/traintest/traintest_herpes_v_scc/squamouscCellCarcinoma/"
    fname = "squamouscCellCarcinoma"
    save_to_dir = "squamouscCellCarcinoma_aug"


    augment(dirName, fname, save_to_dir, count=4)

if True:
    if __name__ == "__main__": main()
else:
    # algorithm to find integer solutions to the following
    # this will make datasets equal
    # x*284 = 242*y | x, y > 15
    for k in range(1,10):
        for i in range(15,40):
            if (i * (284 + k)) % 242 == 0:
                print((i * (284+k)) / 242, f"284 -> {284+k}, i: {i}")
                break
        else:
            continue
        break