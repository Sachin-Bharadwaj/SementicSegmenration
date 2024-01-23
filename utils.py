from typing import List
from PIL import Image
import numpy as np

#def filter_semseg(impath, valid_cls_id:List[int], outpath):
def filter_semseg(args):

    """
    the function take PIL image as input which is segmetation map and filters out class id which are not in valid_cls_id
    im: path for input PIL image
    valid_cls_id: list of valid class ids
    outpath: path where filtered PIL image will be saved
    return: filtered PIL image
    """

    impath, clsnametoid_new, clsnametoid_old, clsnametoid_ignore, outpath, ignore_label = args[0], args[1], args[2], args[3], args[4], args[5]
    im = Image.open(impath)
    image = np.array(im)
    image_ = np.array(image)
    # first set the label of classes not in valid list to ingore_label
    # loop over ignore ids and create ignoremask
    prevmask = False * np.ones_like(image_)
    for cls_name, cls_id in clsnametoid_ignore.items():
        currignoremask = image_ == cls_id
        ignoremask = currignoremask + prevmask
        prevmask = ignoremask
    ignoremask[ignoremask > 0] = True
    image_[ignoremask.astype('bool')] = ignore_label

    for cls_name, cls_id in clsnametoid_old.items():
        # replace the old class labels with new class labels
        image_[image_ == cls_id] = clsnametoid_new[cls_name]

    image = image_.astype('uint8')
    # convert back to PIL image object
    image = Image.fromarray(image)
    # save the image to outpath
    fname = impath.split("\\")[-1]
    image.save(outpath + fname)

