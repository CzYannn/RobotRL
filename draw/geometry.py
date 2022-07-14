


# 2D models

def get_2d_car_model(size):
    # size /= 2
    verts = [
        [-1. * size,  1. * size],  # 矩形左下角的坐标(left,bottom)
        [-1. * size, -1. * size],  # 矩形左上角的坐标(left,top)
        [ 1. * size, -1. * size],  # 矩形右上角的坐标(right,top)
        [ 1. * size,  1. * size],  # 矩形右下角的坐标(right, bottom)
        [-1. * size,  1. * size],  # 封闭到起点
    ]
    return verts

def get_2d_uav_model0(size):
    # size /= 2
    verts = [
        [0., 1. * size],  # 矩形左下角的坐标(left, bottom)
        [-0.5 * size, -0.8 * size],  # 矩形左上角的坐标(left, top)
        [0, -0.2 * size],  # 矩形右上角的坐标(right,top)
        [0.5 * size, -0.8 * size],  # 矩形右下角的坐标(right, bottom)
        [0., 1. * size],  # 封闭到起点
    ]
    return verts

def get_2d_uav_model(size):
    # size /= 2
    verts = [
        [0., 1.*size],  # 矩形左下角的坐标(left, bottom)
        [-1.*size, -1.*size],  # 矩形左上角的坐标(left, top)
        [0., -0.5*size],  # 矩形右上角的坐标(right,top)
        [1.*size, -1.*size],  # 矩形右下角的坐标(right, bottom)
        [0., 1.*size],  # 封闭到起点
    ]
    return verts



# 3D models

def get_car_model(size=0.45, tall=0.05):
    car_model = [[-size, size, tall], [-size, -size, tall], [size, -size, tall], [size, size, tall],
                 [-size, size, 0.0], [-size, -size, 0.0], [size, -size, 0.0], [size, size, 0.0]]
    return car_model

def get_uav_model(size=0.6, tall=0.05):
    uav_model = [[0., size, tall], [-0.5 * size, -0.5 * size, tall], [0., -0.2 * size, tall],
                 [0.5 * size, -0.5 * size, tall],
                 [0., size, -tall], [-0.5 * size, -0.5 * size, -tall], [0., -0.2 * size, -tall],
                 [0.5 * size, -0.5 * size, -tall]]
    return uav_model

def get_building_model(long=10.0, width=10.0, tall=10.0):
    building_model = [[-long / 2, width / 2, tall], [-long / 2, -width / 2, tall], [long / 2, -width / 2, tall],
                      [long / 2, width / 2, tall],
                      [-long / 2, width / 2, 0.0], [-long / 2, -width / 2, 0.0], [long / 2, -width / 2, 0.0],
                      [long / 2, width / 2, 0.0]]
    return building_model

