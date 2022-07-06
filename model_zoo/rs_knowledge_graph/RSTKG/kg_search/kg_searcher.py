import logging
import sys
import time as Time

import numpy as np
sys.path.append('./')
from kg_creater.GLC_loader import get_lon_lat, readTif
from osgeo import gdal
from py2neo import (Graph, Node, NodeMatcher, Relationship,
                    RelationshipMatcher, Subgraph)

g = Graph("http://localhost:7474", auth=("neo4j", "123456"))

def get_semantic(node):
    relamatcher = RelationshipMatcher(g)
    # 首先获取时间状态节点
    relalist = list(relamatcher.match([node]))
    node = relalist[0].end_node
    # 获取类别节点
    relalist = list(relamatcher.match([node]))
    class_value = relalist[0].end_node['value']
    return class_value

# 逆序检索四叉树根节点
def get_upper_layer(checkpoint_node, x, y):
    class_value = 0
    new_checkpoint_node = 0
    relamatcher = RelationshipMatcher(g)
    relalist = list(relamatcher.match([None, checkpoint_node]))
    node = relalist[0].start_node
    if node['UL_x'] >= x and node['DR_x'] < x and node['UL_y'] <= y and node['DR_y'] > y:
        if node['type'] == '真实地表':
            class_value = get_semantic(node)
            new_checkpoint_node = node
        else:
            class_value, new_checkpoint_node = get_next_layer([node], x, y)
    else:
        class_value, new_checkpoint_node = get_upper_layer(node, x, y)
    return class_value, new_checkpoint_node

# 顺序检索四叉树叶子节点
def get_next_layer(upper_node, x, y):
    nodelist = []
    relamatcher = RelationshipMatcher(g)
    relalist = list(relamatcher.match(upper_node))
    for relation in relalist:
        nodelist.append(relation.end_node)

    for node in nodelist:
        if node['UL_x'] >= x and node['DR_x'] < x and node['UL_y'] <= y and node['DR_y'] > y:
            if node['type'] == '真实地表':
                class_value = get_semantic(node)
                checkpoint_node = node
            else:
                class_value, checkpoint_node = get_next_layer([node], x, y)

    return class_value, checkpoint_node

if __name__ == '__main__':
    time_start = Time.time()
    class_value_upper = 0
    checkpoint_node = 0

    fileName = "./image/30-2012-0800-6315-LA93-0M50-E080_re.tif"
    print('正在由影像文件检索时空知识图谱：', fileName)
    image = readTif(fileName)
    im_width = image.RasterXSize # 栅格矩阵的列数   
    im_height = image.RasterYSize # 栅格矩阵的行数
    Geoproj = image.GetProjection() # GeoTiff坐标信息
    GeoTransform = image.GetGeoTransform() # GeoTiff仿射矩阵
    print('GeoTiff坐标信息：', GeoTransform)
    print(f'影像大小为 宽：{im_width} 高：{im_height} ')

    GLC_result = np.full((im_height, im_width), -1)

    # 获取地理单元1
    q = "match(n:地理单元{name:'地理单元1'}) return n"
    nodelist_first = g.run(q)
    for record in nodelist_first:
        node_first = record
    print('已搜索到起始地理单元，开始四叉树检索。')

    for y in range(im_height):
        for x in range(im_width):
            print('\rComplete: {0:.2f}%'.format(((y*im_width) + x + 1)/(im_width*im_height)*100), end='', flush=True)
            if GLC_result[y][x] != -1:
                continue
            GeoX, GeoY = get_lon_lat(GeoTransform, x, y)
            if checkpoint_node and checkpoint_node['UL_x'] >= GeoX and checkpoint_node['DR_x'] < GeoX and checkpoint_node['UL_y'] <= GeoY and checkpoint_node['DR_y'] > GeoY:
                GLC_result[y][x] = class_value_upper
            else:
                if checkpoint_node:
                    class_value_upper, checkpoint_node = get_upper_layer(checkpoint_node, GeoX, GeoY)
                else:
                    class_value_upper, checkpoint_node = get_next_layer(node_first, GeoX, GeoY)
                GLC_result[y][x] = class_value_upper

            for delta in range(1, 60):
                GeoX, GeoY = get_lon_lat(GeoTransform, x, y + delta)
                if y + delta < im_height and checkpoint_node and checkpoint_node['UL_x'] >= GeoX and checkpoint_node['DR_x'] < GeoX and checkpoint_node['UL_y'] <= GeoY and checkpoint_node['DR_y'] > GeoY:
                    GLC_result[y + delta][x] = class_value_upper

    # 创建tif文件
    driver = gdal.GetDriverByName("GTiff")
    New_YG_dataset = driver.Create(r'./result/result.tif', im_width, im_height, 1, gdal.GDT_Byte)
    New_YG_dataset.SetProjection(Geoproj)
    New_YG_dataset.SetGeoTransform(GeoTransform)
    band = New_YG_dataset.GetRasterBand(1)
    band.WriteArray(GLC_result)

    time_end=Time.time()
    total_time = time_end - time_start
    hours = int(total_time/3600)
    minutes = int((total_time%3600)/60)
    seconds = int(total_time%60)
    print(f'\nTime Totally Cost: {hours}h {minutes}m {seconds}s ')

