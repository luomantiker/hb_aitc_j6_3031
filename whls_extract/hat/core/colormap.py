# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np

__all__ = [
    "colors_all",
    "get_colormap_34c",
    "get_colormap_9c_for_bev",
    "get_colormap_for_det_seg",
    "get_colormap_6c_for_bev",
]


def get_colormap_34c():
    colors = np.zeros((256, 1, 3), dtype="uint8")
    colors[0, :, :] = [128, 64, 128]
    colors[1, :, :] = [244, 35, 232]
    colors[2, :, :] = [107, 142, 35]
    colors[3, :, :] = [152, 251, 152]
    colors[4, :, :] = [190, 153, 153]
    colors[5, :, :] = [220, 220, 0]
    colors[6, :, :] = [250, 170, 30]
    colors[7, :, :] = [219, 112, 147]
    colors[8, :, :] = [200, 200, 200]
    colors[9, :, :] = [220, 20, 60]
    colors[10, :, :] = [255, 0, 0]
    colors[11, :, :] = [119, 11, 32]
    colors[12, :, :] = [0, 0, 230]
    colors[13, :, :] = [128, 192, 0]
    colors[14, :, :] = [0, 0, 142]
    colors[15, :, :] = [0, 0, 70]
    colors[16, :, :] = [0, 60, 100]
    colors[17, :, :] = [0, 80, 100]
    colors[18, :, :] = [70, 70, 70]
    colors[19, :, :] = [139, 0, 139]
    colors[20, :, :] = [70, 130, 180]
    colors[21, :, :] = [238, 18, 137]
    colors[22, :, :] = [255, 246, 143]
    colors[23, :, :] = [139, 69, 19]
    colors[24, :, :] = [255, 127, 80]
    colors[25, :, :] = [47, 79, 79]
    colors[26, :, :] = [0, 128, 0]
    colors[27, :, :] = [192, 0, 64]
    colors[28, :, :] = [0, 250, 154]
    colors[29, :, :] = [173, 255, 47]
    colors[30, :, :] = [0, 64, 192]
    colors[31, :, :] = [128, 0, 192]
    colors[32, :, :] = [192, 128, 64]
    colors[33, :, :] = [111, 134, 149]
    return colors.squeeze()


def get_colormap_9c_for_bev():
    colors = np.zeros((256, 1, 3), dtype="uint8")
    colors[0, :, :] = [128, 64, 128]
    colors[1, :, :] = [244, 35, 232]
    colors[2, :, :] = [152, 251, 152]
    colors[3, :, :] = [200, 200, 200]
    colors[4, :, :] = [139, 0, 139]
    colors[5, :, :] = [255, 127, 80]
    colors[6, :, :] = [47, 79, 79]
    colors[7, :, :] = [192, 0, 64]
    colors[8, :, :] = [0, 0, 0]
    return colors.squeeze()


def get_colormap_for_det_seg(cls):  # noqa: D205,D400
    """

    Args:
        cls (str): task type, a str

    Returns (array):bgr color array, shape is Nx3

    """

    colors256 = colors_all()
    if cls == "seg":
        ids = [0, 2, 4, 5, 6, 11, 13, 14, 3, 26, 37, 21, 54, 48, 12, 7]
    elif cls == "det":
        return np.array([[0, 255, 0]])
    else:
        raise NotImplementedError

    rgb_color = colors256[ids, :, :].squeeze()
    bgr_color = rgb_color[:, ::-1]

    return bgr_color


def colors_all():
    colors = np.zeros((256, 1, 3), dtype="uint8")
    # rgb
    colors[0, :, :] = [128, 64, 128]
    colors[1, :, :] = [244, 35, 232]
    colors[2, :, :] = [70, 70, 70]
    colors[3, :, :] = [200, 200, 200]
    colors[4, :, :] = [190, 153, 153]
    colors[5, :, :] = [153, 153, 153]
    colors[6, :, :] = [250, 170, 30]
    colors[7, :, :] = [220, 220, 0]
    colors[8, :, :] = [107, 142, 35]
    colors[9, :, :] = [152, 251, 152]
    colors[10, :, :] = [70, 130, 180]
    colors[11, :, :] = [220, 20, 60]
    colors[12, :, :] = [255, 0, 0]
    colors[13, :, :] = [0, 0, 142]
    colors[14, :, :] = [0, 0, 70]
    colors[15, :, :] = [0, 60, 100]
    colors[16, :, :] = [0, 80, 100]
    colors[17, :, :] = [0, 0, 230]
    colors[18, :, :] = [119, 11, 32]
    colors[19, :, :] = [128, 192, 0]
    colors[20, :, :] = [102, 102, 156]
    colors[21, :, :] = [200, 200, 128]
    colors[22, :, :] = [0, 192, 200]
    colors[23, :, :] = [128, 192, 128]
    colors[24, :, :] = [64, 64, 0]
    colors[25, :, :] = [192, 64, 0]
    colors[26, :, :] = [64, 192, 0]
    colors[27, :, :] = [192, 192, 0]
    colors[28, :, :] = [64, 64, 128]
    colors[29, :, :] = [192, 64, 128]
    colors[30, :, :] = [64, 192, 128]
    colors[31, :, :] = [192, 192, 128]
    colors[32, :, :] = [0, 0, 64]
    colors[33, :, :] = [128, 0, 64]
    colors[34, :, :] = [0, 128, 64]
    colors[35, :, :] = [128, 128, 64]
    colors[36, :, :] = [0, 0, 192]
    colors[37, :, :] = [128, 0, 192]
    colors[38, :, :] = [0, 128, 192]
    colors[39, :, :] = [128, 128, 192]
    colors[40, :, :] = [64, 0, 64]
    colors[41, :, :] = [192, 0, 64]
    colors[42, :, :] = [64, 128, 64]
    colors[43, :, :] = [192, 128, 64]
    colors[44, :, :] = [64, 0, 192]
    colors[45, :, :] = [192, 0, 192]
    colors[46, :, :] = [64, 128, 192]
    colors[47, :, :] = [192, 128, 192]
    colors[48, :, :] = [0, 64, 64]
    colors[49, :, :] = [128, 64, 64]
    colors[50, :, :] = [0, 192, 64]
    colors[51, :, :] = [128, 192, 64]
    colors[52, :, :] = [0, 64, 192]
    colors[53, :, :] = [128, 64, 192]
    colors[54, :, :] = [0, 192, 192]
    colors[55, :, :] = [128, 192, 192]
    colors[56, :, :] = [64, 64, 64]
    colors[57, :, :] = [192, 64, 64]
    colors[58, :, :] = [64, 192, 64]
    colors[59, :, :] = [192, 192, 64]
    colors[60, :, :] = [64, 64, 192]
    colors[61, :, :] = [192, 64, 192]
    colors[62, :, :] = [64, 192, 192]
    colors[63, :, :] = [192, 192, 192]
    colors[64, :, :] = [32, 0, 0]
    colors[65, :, :] = [160, 0, 0]
    colors[66, :, :] = [32, 128, 0]
    colors[67, :, :] = [160, 128, 0]
    colors[68, :, :] = [32, 0, 128]
    colors[69, :, :] = [160, 0, 128]
    colors[70, :, :] = [32, 128, 128]
    colors[71, :, :] = [160, 128, 128]
    colors[72, :, :] = [96, 0, 0]
    colors[73, :, :] = [224, 0, 0]
    colors[74, :, :] = [96, 128, 0]
    colors[75, :, :] = [224, 128, 0]
    colors[76, :, :] = [96, 0, 128]
    colors[77, :, :] = [224, 0, 128]
    colors[78, :, :] = [96, 128, 128]
    colors[79, :, :] = [224, 128, 128]
    colors[80, :, :] = [32, 64, 0]
    colors[81, :, :] = [160, 64, 0]
    colors[82, :, :] = [32, 192, 0]
    colors[83, :, :] = [160, 192, 0]
    colors[84, :, :] = [32, 64, 128]
    colors[85, :, :] = [160, 64, 128]
    colors[86, :, :] = [32, 192, 128]
    colors[87, :, :] = [160, 192, 128]
    colors[88, :, :] = [96, 64, 0]
    colors[89, :, :] = [224, 64, 0]
    colors[90, :, :] = [96, 192, 0]
    colors[91, :, :] = [224, 192, 0]
    colors[92, :, :] = [96, 64, 128]
    colors[93, :, :] = [224, 64, 128]
    colors[94, :, :] = [96, 192, 128]
    colors[95, :, :] = [224, 192, 128]
    colors[96, :, :] = [32, 0, 64]
    colors[97, :, :] = [160, 0, 64]
    colors[98, :, :] = [32, 128, 64]
    colors[99, :, :] = [160, 128, 64]
    colors[100, :, :] = [32, 0, 192]
    colors[101, :, :] = [160, 0, 192]
    colors[102, :, :] = [32, 128, 192]
    colors[103, :, :] = [160, 128, 192]
    colors[104, :, :] = [96, 0, 64]
    colors[105, :, :] = [224, 0, 64]
    colors[106, :, :] = [96, 128, 64]
    colors[107, :, :] = [224, 128, 64]
    colors[108, :, :] = [96, 0, 192]
    colors[109, :, :] = [224, 0, 192]
    colors[110, :, :] = [96, 128, 192]
    colors[111, :, :] = [224, 128, 192]
    colors[112, :, :] = [32, 64, 64]
    colors[113, :, :] = [160, 64, 64]
    colors[114, :, :] = [32, 192, 64]
    colors[115, :, :] = [160, 192, 64]
    colors[116, :, :] = [32, 64, 192]
    colors[117, :, :] = [160, 64, 192]
    colors[118, :, :] = [32, 192, 192]
    colors[119, :, :] = [160, 192, 192]
    colors[120, :, :] = [96, 64, 64]
    colors[121, :, :] = [224, 64, 64]
    colors[122, :, :] = [96, 192, 64]
    colors[123, :, :] = [224, 192, 64]
    colors[124, :, :] = [96, 64, 192]
    colors[125, :, :] = [224, 64, 192]
    colors[126, :, :] = [96, 192, 192]
    colors[127, :, :] = [224, 192, 192]
    colors[128, :, :] = [0, 32, 0]
    colors[129, :, :] = [128, 32, 0]
    colors[130, :, :] = [0, 160, 0]
    colors[131, :, :] = [128, 160, 0]
    colors[132, :, :] = [0, 32, 128]
    colors[133, :, :] = [128, 32, 128]
    colors[134, :, :] = [0, 160, 128]
    colors[135, :, :] = [128, 160, 128]
    colors[136, :, :] = [64, 32, 0]
    colors[137, :, :] = [192, 32, 0]
    colors[138, :, :] = [64, 160, 0]
    colors[139, :, :] = [192, 160, 0]
    colors[140, :, :] = [64, 32, 128]
    colors[141, :, :] = [192, 32, 128]
    colors[142, :, :] = [64, 160, 128]
    colors[143, :, :] = [192, 160, 128]
    colors[144, :, :] = [0, 96, 0]
    colors[145, :, :] = [128, 96, 0]
    colors[146, :, :] = [0, 224, 0]
    colors[147, :, :] = [128, 224, 0]
    colors[148, :, :] = [0, 96, 128]
    colors[149, :, :] = [128, 96, 128]
    colors[150, :, :] = [0, 224, 128]
    colors[151, :, :] = [128, 224, 128]
    colors[152, :, :] = [64, 96, 0]
    colors[153, :, :] = [192, 96, 0]
    colors[154, :, :] = [64, 224, 0]
    colors[155, :, :] = [192, 224, 0]
    colors[156, :, :] = [64, 96, 128]
    colors[157, :, :] = [192, 96, 128]
    colors[158, :, :] = [64, 224, 128]
    colors[159, :, :] = [192, 224, 128]
    colors[160, :, :] = [0, 32, 64]
    colors[161, :, :] = [128, 32, 64]
    colors[162, :, :] = [0, 160, 64]
    colors[163, :, :] = [128, 160, 64]
    colors[164, :, :] = [0, 32, 192]
    colors[165, :, :] = [128, 32, 192]
    colors[166, :, :] = [0, 160, 192]
    colors[167, :, :] = [128, 160, 192]
    colors[168, :, :] = [64, 32, 64]
    colors[169, :, :] = [192, 32, 64]
    colors[170, :, :] = [64, 160, 64]
    colors[171, :, :] = [192, 160, 64]
    colors[172, :, :] = [64, 32, 192]
    colors[173, :, :] = [192, 32, 192]
    colors[174, :, :] = [64, 160, 192]
    colors[175, :, :] = [192, 160, 192]
    colors[176, :, :] = [0, 96, 64]
    colors[177, :, :] = [128, 96, 64]
    colors[178, :, :] = [0, 224, 64]
    colors[179, :, :] = [128, 224, 64]
    colors[180, :, :] = [0, 96, 192]
    colors[181, :, :] = [128, 96, 192]
    colors[182, :, :] = [0, 224, 192]
    colors[183, :, :] = [128, 224, 192]
    colors[184, :, :] = [64, 96, 64]
    colors[185, :, :] = [192, 96, 64]
    colors[186, :, :] = [64, 224, 64]
    colors[187, :, :] = [192, 224, 64]
    colors[188, :, :] = [64, 96, 192]
    colors[189, :, :] = [192, 96, 192]
    colors[190, :, :] = [64, 224, 192]
    colors[191, :, :] = [192, 224, 192]
    colors[192, :, :] = [32, 32, 0]
    colors[193, :, :] = [160, 32, 0]
    colors[194, :, :] = [32, 160, 0]
    colors[195, :, :] = [160, 160, 0]
    colors[196, :, :] = [32, 32, 128]
    colors[197, :, :] = [160, 32, 128]
    colors[198, :, :] = [32, 160, 128]
    colors[199, :, :] = [160, 160, 128]
    colors[200, :, :] = [96, 32, 0]
    colors[201, :, :] = [224, 32, 0]
    colors[202, :, :] = [96, 160, 0]
    colors[203, :, :] = [224, 160, 0]
    colors[204, :, :] = [96, 32, 128]
    colors[205, :, :] = [224, 32, 128]
    colors[206, :, :] = [96, 160, 128]
    colors[207, :, :] = [224, 160, 128]
    colors[208, :, :] = [32, 96, 0]
    colors[209, :, :] = [160, 96, 0]
    colors[210, :, :] = [32, 224, 0]
    colors[211, :, :] = [160, 224, 0]
    colors[212, :, :] = [32, 96, 128]
    colors[213, :, :] = [160, 96, 128]
    colors[214, :, :] = [32, 224, 128]
    colors[215, :, :] = [160, 224, 128]
    colors[216, :, :] = [96, 96, 0]
    colors[217, :, :] = [224, 96, 0]
    colors[218, :, :] = [96, 224, 0]
    colors[219, :, :] = [224, 224, 0]
    colors[220, :, :] = [96, 96, 128]
    colors[221, :, :] = [224, 96, 128]
    colors[222, :, :] = [96, 224, 128]
    colors[223, :, :] = [224, 224, 128]
    colors[224, :, :] = [32, 32, 64]
    colors[225, :, :] = [160, 32, 64]
    colors[226, :, :] = [32, 160, 64]
    colors[227, :, :] = [160, 160, 64]
    colors[228, :, :] = [32, 32, 192]
    colors[229, :, :] = [160, 32, 192]
    colors[230, :, :] = [32, 160, 192]
    colors[231, :, :] = [160, 160, 192]
    colors[232, :, :] = [96, 32, 64]
    colors[233, :, :] = [224, 32, 64]
    colors[234, :, :] = [96, 160, 64]
    colors[235, :, :] = [224, 160, 64]
    colors[236, :, :] = [96, 32, 192]
    colors[237, :, :] = [224, 32, 192]
    colors[238, :, :] = [96, 160, 192]
    colors[239, :, :] = [224, 160, 192]
    colors[240, :, :] = [32, 96, 64]
    colors[241, :, :] = [160, 96, 64]
    colors[242, :, :] = [32, 224, 64]
    colors[243, :, :] = [160, 224, 64]
    colors[244, :, :] = [32, 96, 192]
    colors[245, :, :] = [160, 96, 192]
    colors[246, :, :] = [32, 224, 192]
    colors[247, :, :] = [160, 224, 192]
    colors[248, :, :] = [96, 96, 64]
    colors[249, :, :] = [224, 96, 64]
    colors[250, :, :] = [96, 224, 64]
    colors[251, :, :] = [224, 224, 64]
    colors[252, :, :] = [96, 96, 192]
    colors[253, :, :] = [224, 96, 192]
    colors[254, :, :] = [96, 224, 192]
    colors[255, :, :] = [0, 0, 0]
    return colors


def get_colormap_6c_for_bev():
    colors = np.zeros((256, 1, 3), dtype="uint8")
    colors[0, :, :] = [0, 0, 0]
    colors[1, :, :] = [0, 0, 255]
    colors[2, :, :] = [47, 79, 79]
    colors[3, :, :] = [200, 200, 200]
    colors[4, :, :] = [192, 0, 64]
    colors[5, :, :] = [255, 127, 80]
    return colors.squeeze()
