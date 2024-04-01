import numpy as np
import h5py
import math
import pandas as pd
import numpy as np
import geojson
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import yaml
from shapely.affinity import scale
from matplotlib.patches import PathPatch
from shapely.geometry import LineString, MultiLineString
from geopy.distance import geodesic
from skimage.draw import line as bresenham_line
from shapely.wkt import loads
from shapely.geometry import shape

with open('config.yaml', 'r', encoding='utf-8') as f:
    args = yaml.load(f.read(), Loader=yaml.FullLoader)


def haversine_distance(coord1, coord2):
    """
    使用 Haversine 公式计算两个地点之间的球面距离
    :param coord1: 第一个地点的经纬度，格式为 (latitude, longitude)
    :param coord2: 第二个地点的经纬度，格式为 (latitude, longitude)
    :return: 两个地点之间的距离（单位：千米）
    """
    # 地球半径（单位：千米）
    earth_radius = 6371.0

    # 将经纬度转换为弧度
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    # 计算经纬度差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # 使用 Haversine 公式计算距离
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 计算距离
    distance = earth_radius * c

    return distance


def generate_matrix(datas, scal=1000):
    geome_file = f"{args['data']}graph_sensor_locations.csv"
    genme_list = pd.read_csv(geome_file)
    geojson_file = f"{args['data']}METR-LA.geojson"
    gdf = gpd.read_file(geojson_file)

    # 获取最大最小经纬度
    min_lat, min_lon = gdf.geometry.bounds[['miny', 'minx']].min()
    max_lat, max_lon = gdf.geometry.bounds[['maxy', 'maxx']].max()

    # 扩大1000倍
    min_lat *= scal
    min_lon *= scal
    max_lat *= scal
    max_lon *= scal

    # 计算矩阵的大小
    matrix_width = int((max_lon - min_lon)) + 1
    matrix_height = int((max_lat - min_lat)) + 1

    print(f"matrix_width:{matrix_width}, matrix_height:{matrix_height}")

    # 创建图形和轴
    fig, ax = plt.subplots()

    # 绘制线几何
    gdf.plot(ax=ax)
    result_matrixs = []
    sensor_pos = {}
    for data in datas:
        # 创建矩阵，并将坐标范围内每条线上的位置对应的值设置为当前线的编号
        result_matrix = np.zeros((matrix_height, matrix_width))
        # 获取唯一的线编号
        line_numbers = np.arange(0, len(gdf))
        for line_number, line in zip(line_numbers, gdf['geometry']):
            if gdf['sensor'][line_number] is not None:
                sensors = gdf['sensor'][line_number].split(",")
            else:
                continue
            if isinstance(line, MultiLineString):
                line = str(line).split("((")[1].strip("))").split(",")
                line_list = []
                for single_line in line:
                    line_list.append(list(map(float, single_line.split())))

                line_coords = np.array(line_list) * scal  # 扩大1000倍，并转置
                coordinates = line_coords / scal
            else:
                line_coords = np.array(line.xy).T * scal  # 扩大1000倍，并转置
                linestring_geometry = shape(line)
                coordinates = list(linestring_geometry.coords)

            # 将坐标映射到矩阵中
            mapped_coords = np.floor(line_coords - np.array([min_lon, min_lat])).astype(int)

            sensors = [int(x) for x in sensors]
            rrs = []
            ccs = []
            for i in range(mapped_coords.shape[0] - 1):
                rr, cc = bresenham_line(mapped_coords[i, 1], mapped_coords[i, 0], mapped_coords[i + 1, 1],
                                        mapped_coords[i + 1, 0])
                rrs.append(rr)
                ccs.append(cc)
            rrs = [item for sublist in rrs for item in sublist]
            ccs = [item for sublist in ccs for item in sublist]
            n = len(rrs)
            deta_v = 0
            if len(sensors) > 1:
                latitude_sensors = [genme_list["latitude"][int(x)] for x in sensors]
                longitude_sensors = [genme_list["longitude"][int(x)] for x in sensors]
                geo_sensors = list(zip(latitude_sensors, longitude_sensors))

                way1 = haversine_distance((coordinates[0][1], coordinates[0][0]), geo_sensors[0])
                way2 = haversine_distance((coordinates[0][1], coordinates[0][0]), geo_sensors[1])
                way = [way1, way2]
                min_value = min(way)
                min_index = way.index(min_value)
                if min_index != 0:
                    sensors.reverse()
                v_list = data[sensors, 0]
                deta_v = v_list[1] - v_list[0]
                if n > 1:
                    deta_v = deta_v / (n - 1)
                sensor_pos.update({sensors[0]: [rrs[0], ccs[0]], sensors[1]: [rrs[-1], ccs[-1]]})
                v = v_list[0]
            else:
                v_list = data[sensors][0]
                v = v_list[0]
                sensor_pos.update({sensors[0]: [rrs[0], ccs[0]]})
            for r in range(0, n):
                if result_matrix[rrs[r], ccs[r]] == 0:
                    result_matrix[rrs[r], ccs[r]] = v
                else:
                    result_matrix[rrs[r], ccs[r]] = (result_matrix[rrs[r], ccs[r]] + v) / 2
                v += deta_v
            result_matrix[rrs[-1], ccs[-1]] = v_list[-1]
        result_matrixs.append(result_matrix)
    result_matrixs = np.array(result_matrixs, dtype=np.float32)
    sensor_pos = {key: value for key, value in sorted(sensor_pos.items())}
    return result_matrixs, sensor_pos


def generate_data_date(datas):
    sample_num = datas.shape[0]
    x_data = []
    time_step = args["time_step"]
    time_end = args['time_end']
    for i in range(sample_num - time_step - time_end + 1):
        x_data.append(datas[i:i + time_step, 0, 1:])
    x_data = np.asarray(x_data, dtype='float32')
    del datas
    sample_num = x_data.shape[0]
    num_test = round(sample_num * 0.2)
    num_train = round(sample_num * 0.7)
    num_val = sample_num - num_test - num_train
    x_train = x_data[:num_train]
    x_val = x_data[num_train:num_train + num_val]
    x_test = x_data[-num_test:]
    np.savez(f"{args['data']}/train_date.npz", x=x_train)
    np.savez(f"{args['data']}/val_date.npz", x=x_val)
    np.savez(f"{args['data']}/test_date.npz", x=x_test)
    print("生成时间数据成功")


def generate_data(matrix, sensors, datas):
    sample_num = matrix.shape[0]
    x_data = []
    time_step = args["time_step"]
    time_end = args['time_end']
    sensor_list = np.sort(np.int16(np.array(list(sensors.keys()))))
    for i in range(sample_num - time_step - time_end + 1):
        x_data.append(matrix[i:i + time_step])
    x_data = np.asarray(x_data, dtype='float32')
    del matrix

    data_y = np.asarray(datas).astype('float32')
    y_data = []
    for i in range(sample_num - time_step - time_end + 1):
        y_data.append(data_y[i + time_step:i + time_step + time_end][:, sensor_list, 0])
    y_data = np.asarray(y_data).astype('float32')
    del data_y

    print('x_data.shape:', x_data.shape)
    print(f"y_data.shape:{y_data.shape}")

    sample_num = x_data.shape[0]
    num_test = round(sample_num * 0.2)
    num_train = round(sample_num * 0.7)
    num_val = sample_num - num_test - num_train
    x_train = x_data[:num_train]
    x_val = x_data[num_train:num_train + num_val]
    x_test = x_data[-num_test:]
    del x_data

    y_train = y_data[:num_train]
    y_val = y_data[num_train:num_train + num_val]
    y_test = y_data[-num_test:]
    del y_data

    train = {'x': x_train, 'y': y_train}
    np.savez(f"{args['data']}/train.npz", **train)
    del train
    val = {'x': x_val, 'y': y_val}
    np.savez(f"{args['data']}/val.npz", **val)
    del val
    test = {'x': x_test, 'y': y_test}
    np.savez(f"{args['data']}/test.npz", **test)
    del test


def main():
    print("start")
    datas = np.load(f"{args['data']}data.npz")["data"].astype(np.float32)
    generate_data_date(datas)
    result_matrixs, sensor_pos = generate_matrix(datas)
    generate_data(result_matrixs, sensor_pos, datas)
    print("finish")


if __name__ == '__main__':
    main()
