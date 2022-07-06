from osgeo import gdal
import numpy as np
from py2neo import Graph, Node, Relationship, NodeMatcher, Subgraph
import time as Time

g = Graph("http://localhost:7474", auth=("neo4j", "123456"))
CLASS_LIST = ['No Value/无值', 
              'Cultivated Land/耕地',
              'Forest/林地',
              'Grass Land/草地', 
              'Shrubland/灌木地', 
              'Wetland/湿地', 
              'Water Body/水体', 
              'Tundra/苔原', 
              'Artificial Surfaces/人造地表', 
              'Bareland/裸地', 
              'Permanent Snow and Ice/冰川和永久积雪',
              'Sea/海水']

# 读取tif数据
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset

# 像素坐标到地理坐标仿射变换
def CoordTransf(Xpixel, Ypixel, GeoTransform):
    YGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2]
    XGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5]
    return XGeo, YGeo

# 可以不写,没啥用
def get_lon_lat(GeoTransform, Xpixel, Ypixel):
    XGeo, YGeo = CoordTransf(Xpixel, Ypixel, GeoTransform)
    return XGeo, YGeo

# py2neo版本
def create_node(x_start, y_start, x_end, y_end, GeoTransform, count, layer, data_store=False):
    global checkpoint
    UL_x, UL_y = get_lon_lat(GeoTransform, x_start - 0.5, y_start - 0.5)
    DR_x, DR_y = get_lon_lat(GeoTransform, x_end + 0.5, y_end + 0.5)

    if data_store==False:
        node_type = '索引'
    else:
        node_type = '真实地表'

    node = Node('地理单元', name='地理单元' + str(count),
                                UL_x=UL_x, UL_y=UL_y,
                                DR_x=DR_x, DR_y=DR_y,
                                type=node_type,
                                layer=layer,
                                Coordinate='EPSG:4326')

    count = count + 1
    return count, node


def spatial_quad_tree_first(GLC_image, data_type, time, count=1):
    layer = 1
    coordinate_list = []
    GeoTransform = GLC_image.GetGeoTransform()
    im_width = GLC_image.RasterXSize #栅格矩阵的列数   
    im_height = GLC_image.RasterYSize #栅格矩阵的行数
    global im_data_ori
    im_data_ori = GLC_image.ReadAsArray(0, 0, im_width, im_height) #获取数据
    global last_print
    last_print = 0
    global relation_all
    global node_all
    global checkpoint
    relation_all = []
    node_all = []

    # 创建语义类别节点
    for class_type in CLASS_LIST:
        matcher = NodeMatcher(g)
        nodelist = list(matcher.match('类别语义', name=class_type))
        if len(nodelist) == 0:
            type_node = Node('类别语义', name=class_type)
            g.create(type_node)

    if len(np.unique(im_data_ori)) == 1:
        # 建立实际地物节点
        count, now_node = create_node(0, 0,im_width, im_height, GeoTransform, count, layer, data_store=True)
        if count > checkpoint + 1:
            node_all.append(now_node)
            # 建立时间状态节点
            time_node = Node('时间状态', name='地理单元' + str(count-1) + '_' + time, type=data_type, time=time)
            node_all.append(time_node)
            n2t = Relationship(now_node, '时间状态', time_node)
            relation_all.append(n2t)

            if int(np.unique(im_data_ori)[0]) == 255:
                class_type = CLASS_LIST[11]
            else:
                class_type = CLASS_LIST[int(np.unique(im_data_ori)[0]/10)]

            # 匹配或建立类别语义节点
            matcher = NodeMatcher(g)
            nodelist = list(matcher.match('类别语义', name=class_type))
            type_node = nodelist[0]
            node_all.append(type_node)
            t2t = Relationship(time_node, '地表类别', type_node)
            relation_all.append(t2t)
    else:
        # 建立索引节点
        if count > checkpoint:
            count, now_node = create_node(0, 0,im_width, im_height, GeoTransform, count, layer)
            node_all.append(now_node)
        else:
            now_node = count
            count = count + 1
        # 空间四叉树划分
        width_new = int((im_width - 1)/2)
        height_new = int((im_height - 1)/2)
        coordinate_list.append([0, 0, width_new, height_new])
        coordinate_list.append([width_new + 1, 0, im_width - 1, height_new])
        coordinate_list.append([0, height_new + 1, width_new, im_height - 1])
        coordinate_list.append([width_new + 1, height_new + 1, im_width - 1, im_height - 1])
        layer = layer + 1

        # 对子节点进行构造
        for i in range(len(coordinate_list)):
            count = spatial_quad_tree(coordinate_list[i][0], coordinate_list[i][1], coordinate_list[i][2], coordinate_list[i][3], GeoTransform, count, now_node, layer, time, data_type)

    if len(node_all) > 0 or len(relation_all) > 0: 
        subgraph = Subgraph(node_all, relation_all)
        g.create(subgraph)
    return count


def spatial_quad_tree(x_start, y_start, x_end, y_end, GeoTransform, count, upper_node, layer, time, data_type):
    coordinate_list = []
    global complete_area
    global im_data_ori
    global relation_all
    global node_all
    global last_print
    global checkpoint
    if y_start < y_end and x_start < x_end:
        im_data = im_data_ori[y_start:y_end, x_start:x_end]
    elif y_start < y_end:
        im_data = im_data_ori[y_start:y_end, x_start]
    elif x_start < x_end:
        im_data = im_data_ori[y_start, x_start:x_end]
    else:
        im_data = im_data_ori[y_start, x_start]

    
    if len(np.unique(im_data)) == 1:
        complete_area = complete_area + (x_end - x_start + 1)*(y_end - y_start + 1)
        now_print = complete_area/all_area*100
        if (now_print - last_print) > 0.01:
            print('\rComplete: {0:.2f}%'.format(now_print), end='', flush=True)
            last_print = now_print
        if count > checkpoint:
            # 建立实际地物节点
            count, now_node = create_node(x_start, y_start, x_end, y_end, GeoTransform, count, layer, data_store=True)
            node_all.append(now_node)
            if str(type(upper_node)).split("'")[1] == 'int':
                matcher = NodeMatcher(g)
                nodelist = list(matcher.match('地理单元', name='地理单元' + str(upper_node)))
                upper_node = nodelist[0]
                node_all.append(upper_node)
            # 建立上下级关系
            u2n = Relationship(upper_node, '叶子节点', now_node)
            relation_all.append(u2n)
            # 建立时间状态节点
            time_node = Node('时间状态', name='地理单元' + str(count-1) + '_' + time, type=data_type, time=time)
            node_all.append(time_node)
            n2t = Relationship(now_node, '时间状态', time_node)
            relation_all.append(n2t)

            if int(np.unique(im_data)[0]) == 255:
                class_type = CLASS_LIST[11]
            else:
                class_type = CLASS_LIST[int(np.unique(im_data)[0]/10)]

            # 匹配或建立类别语义节点
            matcher = NodeMatcher(g)
            nodelist = list(matcher.match('类别语义', name=class_type))
            type_node = nodelist[0]
            node_all.append(type_node)
            t2t = Relationship(time_node, '地表类别', type_node)
            relation_all.append(t2t)

            if len(relation_all) > 1000000:
                subgraph = Subgraph(node_all, relation_all)
                g.create(subgraph)
                relation_all = []
                node_all = []
        else:
            count = count + 1
    else:
        # 建立索引节点
        if count > checkpoint:
            count, now_node = create_node(x_start, y_start, x_end, y_end, GeoTransform, count, layer)
            node_all.append(now_node)
            if str(type(upper_node)).split("'")[1] == 'int':
                matcher = NodeMatcher(g)
                nodelist = list(matcher.match('地理单元', name='地理单元' + str(upper_node)))
                upper_node = nodelist[0]
                node_all.append(upper_node)
            # 建立上下级关系
            u2n = Relationship(upper_node, '叶子节点', now_node)
            relation_all.append(u2n)
        else:
            now_node = count
            count = count + 1
        # 空间四叉树划分
        if y_start < y_end and x_start < x_end:
            width_new = int((x_start + x_end)/2)
            height_new = int((y_start + y_end)/2)
            coordinate_list.append([x_start, y_start, width_new, height_new])
            coordinate_list.append([width_new + 1, y_start, x_end, height_new])
            coordinate_list.append([x_start, height_new + 1, width_new, y_end])
            coordinate_list.append([width_new + 1, height_new + 1, x_end, y_end])
        elif y_start < y_end:
            height_new = int((y_start + y_end)/2)
            coordinate_list.append([x_start, y_start, x_end, height_new])
            coordinate_list.append([x_start, height_new + 1, x_end, y_end])
        elif x_start < x_end:
            width_new = int((x_start + x_end)/2)
            coordinate_list.append([x_start, y_start, width_new, y_end])
            coordinate_list.append([width_new + 1, y_start, x_end, y_end])
        
        layer = layer + 1

        # 对子节点进行构造
        for i in range(len(coordinate_list)):
            count = spatial_quad_tree(coordinate_list[i][0], coordinate_list[i][1], coordinate_list[i][2], coordinate_list[i][3], GeoTransform, count, now_node, layer, time, data_type)

    return count


if __name__ == '__main__':
    fileName = "./Globeland30-2020/France_2020_GLC.tif"
    data_type = 'Globeland30'
    time = '2020'
    print('正在由文件创建时空知识图谱：', fileName)
    GLC_image = readTif(fileName)
    im_width = GLC_image.RasterXSize #栅格矩阵的列数   
    im_height = GLC_image.RasterYSize #栅格矩阵的行数
    GeoTransform = GLC_image.GetGeoTransform() #GeoTiff坐标信息
    print('GeoTiff坐标信息：', GeoTransform)
    complete_area = 0
    all_area = im_width*im_height
    checkpoint = 0
    print('Create KG:')
    time_start = Time.time()
    count = spatial_quad_tree_first(GLC_image, data_type, time)
    print('\nTotal nodes:', count - 1)
    time_end=Time.time()
    print('Time Totally Cost', time_end - time_start)


