from annofile_process import integrate_annofile,concat_annotation
from eyes_extraction import extracteye_gazefollow,extracteye_videotargetatt
from depth_estimation import depgen_gazefollow,depgen_videotargetatt





if __name__ == '__main__':

    # deal with the annotation file
    integrate_annofile(dataset="gazefollow", type="train")
    integrate_annofile(dataset="gazefollow", type="test")

    integrate_annofile(dataset="videoatt", type="train")
    integrate_annofile(dataset="videoatt", type="test")

    # Crop the eye coordinate from the scene img
    # On gazefollow dataset
    extracteye_gazefollow("train")
    extracteye_gazefollow("test")

    concat_annotation(dataset="gazefollow",type="train")
    concat_annotation(dataset="gazefollow",type="test")

    extracteye_videotargetatt("train")
    extracteye_videotargetatt("test")

    concat_annotation("videoatt",type="train")
    concat_annotation("videoatt",type="test")

    # Generate the depth image from the scene img
    # On gazefollow dataset
    depgen_gazefollow()

    # On videoattentiontarget
    depgen_videotargetatt()

