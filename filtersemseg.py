import concurrent.futures
from utils import filter_semseg
from path import Path
import glob
from torchvision.datasets import Cityscapes

if __name__ == "__main__":
    rootdir = "C:/Sachin/CVAIAC2023/synscapes/Synscapes/img/semseg"
    outdir = "C:/Sachin/CVAIAC2023/synscapes/Synscapes/img/semseg_filt/"
    # get list of all seg images
    flist_seg = glob.glob(rootdir + "/*.png")
    #valid cls ids
    cls_name_id = [(clsdesc.name, clsdesc.id) for clsdesc in Cityscapes.classes if not clsdesc.ignore_in_eval]
    cls_name = [clsdesc.name for clsdesc in Cityscapes.classes if not clsdesc.ignore_in_eval]
    cls_id = [clsdesc.id for clsdesc in Cityscapes.classes if not clsdesc.ignore_in_eval]
    cls_name_id_ignore = [(clsdesc.name, clsdesc.id) for clsdesc in Cityscapes.classes if clsdesc.ignore_in_eval]

    clsnametoid_new = {clsname: i for i, clsname in enumerate(cls_name)}
    clsnametoid_old = {clsname: clsid for clsname, clsid in cls_name_id}
    clsnametoid_ignore = {clsname: clsid for clsname, clsid in cls_name_id_ignore}
    ignore_label = 255

    args_ = [(fpath, clsnametoid_new, clsnametoid_old, clsnametoid_ignore, outdir, ignore_label) for fpath in flist_seg]
    #print(args_)
    #list(map(filter_semseg, args_))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(filter_semseg, args_))