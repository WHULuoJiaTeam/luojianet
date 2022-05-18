from platform import node
from tkinter.tix import Tree
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy as np
from py2neo import (Graph, Node, NodeMatcher, Relationship,
                    RelationshipMatcher, Subgraph)
import time as Time
import os
from tqdm import tqdm
import sys
sys.path.append('./')
from kg_creater.GLC_loader import readTif, CoordTransf
g = Graph("http://localhost:7474", auth=("neo4j", "123456"))
from PIL import Image


corr={
    'forest':10,
    'park':4,
    'residential':1,
    'industrial':2,
    'cemetery':2,
    'allotments':5,
    'meadow':7,
    'commercial':2,
    'nature_reserve':12,
    'recreation_ground':4,
    'retail':2,
    'military':2,
    'quarry':3,
    'orchard':6,
    'vineyard':6,
    'scrub':11,
    'grass':7,
    'heath':12,
    'farmyard':5,
    'farmland':5
}

palette=[ [35, 31, 32],
    [219, 95, 87],
    [219, 151, 87],
    [219, 208, 87], 
    [173, 219, 87], 
    [117, 219, 87],
    [123, 196, 123],
    [88, 177, 88],
    [212,246,212],
    [176,226,176],
    [0, 128, 0],
    [88, 176, 167],
    [153, 93, 19],
    [87, 155, 219],
    [0, 98, 255],
    [35, 31, 32] ]

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]

def readshp(filePath):
    # 设置环境
    gdal.SetConfigOption('GDAL_FILENAME_IS_UTF8', 'YES')
    gdal.SetConfigOption('SHAPE_ENCODING', 'UTF-8')
    ogr.RegisterAll()

    # 打开文件
    ds = ogr.Open(filePath, 0)
    data = []
    if ds == None:
        print('打开文件失败！')
    else:
        # 获取图层数
        iLayerCount = ds.GetLayerCount()
        print('共有图层：', iLayerCount)

        # 读取图层
        oLayer = ds.GetLayerByIndex(0)
        if oLayer == None:
            print('获取图层失败！')

        # 获取坐标系统
        srs = oLayer.GetSpatialRef()

        # 获取要素个数
        oLayer.ResetReading()
        num = oLayer.GetFeatureCount(0)
        print('共有要素：', num)

        with tqdm(total=num - 1, desc='要素提取已完成', mininterval=0.2) as pbar:
            for i in range(1, num):
                # 读取要素类别及坐标信息
                ofeature = oLayer.GetFeature(i)
                name = ofeature.GetFieldAsString('fclass')
                geom = str(ofeature.GetGeometryRef())
                data.append([i, name, geom])
                pbar.update(1)

    return data, srs

def writeshp(filePath, data, srs):
    data.sort(key=lambda x:x[0])
    # 设置环境
    gdal.SetConfigOption('GDAL_FILENAME_IS_UTF8', 'YES')
    gdal.SetConfigOption('SHAPE_ENCODING', 'UTF-8')
    ogr.RegisterAll()

    # 调用驱动
    strDriverName = 'ESRI Shapefile'
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        return "驱动不可用！"

    oDS = oDriver.CreateDataSource(filePath)
    if oDS == None:
        return "创建文件失败！"

    papszLCO = []
    geosrs = osr.SpatialReference()
    geosrs.ImportFromWkt(str(srs))
    
    ogr_type = ogr.wkbMultiPolygon
    oLayer = oDS.CreateLayer('MultiPolygon', geosrs, ogr_type, papszLCO)
    if oLayer == None:
        return "图层创建失败！"
    
    oID = ogr.FieldDefn('fid', ogr.OFTInteger)
    oLayer.CreateField(oID, 1)
    oClass = ogr.FieldDefn('fclass', ogr.OFTString)
    oLayer.CreateField(oClass, 1)

    oDefn = oLayer.GetLayerDefn()

    with tqdm(total=len(data), desc='要素写入已完成', mininterval=0.2) as pbar:
        for index, feature in enumerate(data):
            oFeature = ogr.Feature(oDefn)
            oFeature.SetField('fid', feature[0])
            oFeature.SetField('fclass', feature[1])
            geom = ogr.CreateGeometryFromWkt(feature[2])
            oFeature.SetGeometry(geom)
            oLayer.CreateFeature(oFeature)
            pbar.update(1)
    
    return 'Success'

def writetif(name, data, srs):
    data.sort(key=lambda x:x[0])
    # 设置环境
    gdal.SetConfigOption('GDAL_FILENAME_IS_UTF8', 'YES')
    gdal.SetConfigOption('SHAPE_ENCODING', 'UTF-8')
    ogr.RegisterAll()

    # 调用驱动
    strDriverName = 'ESRI Shapefile'
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        return "驱动不可用！"

    oDS = oDriver.CreateDataSource('tmp.shp')
    if oDS == None:
        return "创建文件失败！"

    papszLCO = []
    geosrs = osr.SpatialReference()
    geosrs.ImportFromWkt(str(srs))
    
    ogr_type = ogr.wkbMultiPolygon
    oLayer = oDS.CreateLayer('MultiPolygon', geosrs, ogr_type, papszLCO)
    if oLayer == None:
        return "图层创建失败！"
    
    oID = ogr.FieldDefn('fid', ogr.OFTInteger)
    oLayer.CreateField(oID, 1)
    oClass = ogr.FieldDefn('type', ogr.OFTString)
    oLayer.CreateField(oClass, 1)

    oDefn = oLayer.GetLayerDefn()

    with tqdm(total=len(data), desc='要素写入已完成', mininterval=0.2) as pbar:
        for index, feature in enumerate(data):
            oFeature = ogr.Feature(oDefn)
            oFeature.SetField('fid', feature[0])
            oFeature.SetField('type', corr[feature[1]])
            geom = ogr.CreateGeometryFromWkt(feature[2])
            oFeature.SetGeometry(geom)
            oLayer.CreateFeature(oFeature)
            pbar.update(1)

    outputFileName='OSM/'+name
    field='type'

    x_min=int(name.split('-')[2])*1000
    y_min=int(name.split('-')[3])*1000
    pixel_width=0.5
    x_res = 10000
    y_res = 10000
    target_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, x_res, y_res, 1, gdal.GDT_Byte,options=["TILED=YES","COMPRESS=DEFLATE"])
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, -1 * pixel_width))
    band = target_ds.GetRasterBand(1)
    NoData_value = -999
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], oLayer, options=["ATTRIBUTE=" + field])

    seg_img = Image.fromarray(target_ds.ReadAsArray()).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    seg_img.save(outputFileName.replace('tif','png'))

    target_ds = None
    oLayer=None
    oDS=None
    print(name)

def createIndex(upper_node, layer, count, bottom_node_list, node_list, rela_list):
    gamma = 100000
    new_x = (upper_node['UL_x'] + upper_node['DR_x'])/2
    new_y = (upper_node['UL_y'] + upper_node['DR_y'])/2
    node_type = '根索引'
    if (upper_node['DR_x'] - upper_node['UL_x'])/2 < gamma and (upper_node['UL_y'] - upper_node['DR_y'])/2 < gamma:
        node_type = '叶子索引'

    # 左上
    node = Node('地理索引', name='地理索引' + str(count), layer=layer, UL_x=upper_node['UL_x'], UL_y=upper_node['UL_y'], DR_x=new_x, DR_y=new_y, type=node_type)
    rela = Relationship(upper_node, '下级索引', node)
    node_list.append(node)
    rela_list.append(rela)
    count = count + 1
    if (upper_node['DR_x'] - upper_node['UL_x'])/2 > gamma and (upper_node['UL_y'] - upper_node['DR_y'])/2 > gamma:
        bottom_node_list, node_list, rela_list, count = createIndex(node, layer + 1, count, bottom_node_list, node_list, rela_list)
    else:
        bottom_node_list.append(node)
    
    # 右上
    node = Node('地理索引', name='地理索引' + str(count), layer=layer, UL_x=new_x, UL_y=upper_node['UL_y'], DR_x=upper_node['DR_x'], DR_y=new_y, type=node_type)
    rela = Relationship(upper_node, '下级索引', node)
    node_list.append(node)
    rela_list.append(rela)
    count = count + 1
    if (upper_node['DR_x'] - upper_node['UL_x'])/2 > gamma and (upper_node['UL_y'] - upper_node['DR_y'])/2 > gamma:
        bottom_node_list, node_list, rela_list, count = createIndex(node, layer + 1, count, bottom_node_list, node_list, rela_list)
    else:
        bottom_node_list.append(node)

    # 左下
    node = Node('地理索引', name='地理索引' + str(count), layer=layer, UL_x=upper_node['UL_x'], UL_y=new_y, DR_x=new_x, DR_y=upper_node['DR_y'], type=node_type)
    rela = Relationship(upper_node, '下级索引', node)
    node_list.append(node)
    rela_list.append(rela)
    count = count + 1
    if (upper_node['DR_x'] - upper_node['UL_x'])/2 > gamma and (upper_node['UL_y'] - upper_node['DR_y'])/2 > gamma:
        bottom_node_list, node_list, rela_list, count = createIndex(node, layer + 1, count, bottom_node_list, node_list, rela_list)
    else:
        bottom_node_list.append(node)

    # 右下
    node = Node('地理索引', name='地理索引' + str(count), layer=layer, UL_x=new_x, UL_y=new_y, DR_x=upper_node['DR_x'], DR_y=upper_node['DR_y'], type=node_type)
    rela = Relationship(upper_node, '下级索引', node)
    node_list.append(node)
    rela_list.append(rela)
    count = count + 1
    if (upper_node['DR_x'] - upper_node['UL_x'])/2 > gamma and (upper_node['UL_y'] - upper_node['DR_y'])/2 > gamma:
        bottom_node_list, node_list, rela_list, count = createIndex(node, layer + 1, count, bottom_node_list, node_list, rela_list)
    else:
        bottom_node_list.append(node)

    return bottom_node_list, node_list, rela_list, count

# 顺序检索四叉树叶子节点
def get_next_layer(upper_node, x, y):
    nodelist = []
    relamatcher = RelationshipMatcher(g)
    relalist = list(relamatcher.match(upper_node))
    for relation in relalist:
        nodelist.append(relation.end_node)

    for node in nodelist:
        if node['UL_x'] < x and node['DR_x'] > x and node['UL_y'] > y and node['DR_y'] < y:
            if node['type'] == '叶子索引':
                checkpoint_node = node
            else:
                checkpoint_node = get_next_layer([node], x, y)

    return checkpoint_node

def createIndex_main(srs):
    node_list = []
    rela_list = []
    bottom_node_list = []
    count = 1
    layer = 1
    node0 = Node('矢量数据', name='OSM', srs = str(srs))
    node1 = Node('地理索引', name='地理索引' + str(count), layer=layer, UL_x=96599.0469, UL_y=7111084.0000, DR_x=1242385.8750, DR_y=6040766.0000)
    rela0 = Relationship(node0, '包含索引', node1)
    node_list.append(node0)
    node_list.append(node1)
    rela_list.append(rela0)
    subgraph = Subgraph(node_list, rela_list)
    g.create(subgraph)
    node_list = []
    rela_list = []
    layer =layer + 1
    count = count + 1

    bottom_node_list, node_list, rela_list, count = createIndex(node1, layer, count, bottom_node_list, node_list, rela_list)
    print('叶子索引节点共计：', len(bottom_node_list))
    subgraph = Subgraph(node_list, rela_list)
    g.create(subgraph)

def createKG(data_list, srs, createIndex=False, add_node=False):
    if createIndex == True:
        createIndex_main(srs)
    
    node_list = []
    rela_list = []
    bottom_node = 0

    matcher = NodeMatcher(g)
    node0 = list(matcher.match('矢量数据', name='OSM'))[0]
    relamatcher = RelationshipMatcher(g)
    rela = list(relamatcher.match([node0]))[0]
    node1 = rela.end_node

    with tqdm(total=len(data_list), desc='要素入库已完成', mininterval=0.2) as pbar:
        for (i, data) in enumerate(data_list):
            if add_node == True:
                node_get = 'match (n:矢量要素{name:"' + data[1] + '", data:"' + data[2] + '", type:"' + str(data[2]).split(' ')[0] + '", fid:' + str(data[0]) + '}) return n'
                node_get_list = g.run(node_get).data()
                if len(node_get_list) > 0:
                    node = node_get_list[0]['n']
                else:
                    node = Node('矢量要素', name=data[1], data=data[2], type=str(data[2]).split(' ')[0], fid=data[0])
                    node_list.append(node)
            else:
                node = Node('矢量要素', name=data[1], data=data[2], type=str(data[2]).split(' ')[0], fid=data[0])
                node_list.append(node)

            linked_node_list = []
            polygon_list_2 = data[2].split('(((')[-1].split(')))')[0].split(')),((')
            for polygon_2 in polygon_list_2:
                polygon_list = polygon_2.split('),(')
                for polygon in polygon_list:
                    coordinate_list = polygon.split(',')
                    for coordinate in coordinate_list:
                        x = float(coordinate.split(' ')[0])
                        y = float(coordinate.split(' ')[1])
                        if bottom_node == 0 or (bottom_node['UL_x'] > x or bottom_node['DR_x'] < x or bottom_node['UL_y'] < y or bottom_node['DR_y'] > y):
                            bottom_node = get_next_layer([node1], x, y)
                        if bottom_node not in linked_node_list:
                            rela = Relationship(bottom_node, '包含', node)
                            rela_list.append(rela)
                            linked_node_list.append(bottom_node)
    
            if len(rela_list) > 1000:
                subgraph = Subgraph(node_list, rela_list)
                g.create(subgraph)
                rela_list = []

            pbar.update(1)

    subgraph = Subgraph(node_list, rela_list)
    g.create(subgraph)
    return 1

def get_shapes(data_list, bottom_node):
    # relamatcher = RelationshipMatcher(g)
    # rela_list = list(relamatcher.match([bottom_node]))

    # with tqdm(total=len(rela_list), desc='要素检索已完成') as pbar:
    #     for rela in rela_list:
    #         fclass = rela.end_node['name']
    #         data = rela.end_node['data']
    #         if [fclass, data] not in data_list:
    #             data_list.append([fclass, data])
    #         pbar.update(1)

    rela_get = 'match (n:地理索引)-[r:包含]->(q) where n.name="' + str(bottom_node['name']) + '" return q'
    node_list = g.run(rela_get)

    for node in node_list:
        fid = int(node['q']['fid'])
        fclass = node['q']['name']
        data = node['q']['data']
        if [fid, fclass, data] not in data_list:
            data_list.append([fid, fclass, data])

    return data_list

def searchKG(imagePath):
    image = readTif(imagePath)
    im_width = image.RasterXSize #栅格矩阵的列数   
    im_height = image.RasterYSize #栅格矩阵的行数
    GeoTransform = image.GetGeoTransform() #GeoTiff坐标信息
    print('GeoTiff坐标信息：', GeoTransform)

    UL_y, UL_x = CoordTransf(0, 0, GeoTransform)
    DR_y, DR_x = CoordTransf(im_width, im_height, GeoTransform)

    data_list = []
    bottom_node_list = []
    Coordinate_list = [[UL_x, UL_y], [UL_x, DR_y], [DR_x, UL_y], [DR_x, DR_y]]
    matcher = NodeMatcher(g)
    node0 = list(matcher.match('矢量数据', name='OSM'))[0]
    srs = node0['srs']
    relamatcher = RelationshipMatcher(g)
    rela = list(relamatcher.match([node0]))[0]
    node1 = rela.end_node

    for coordinate in Coordinate_list:
        bottom_node = get_next_layer([node1], coordinate[0], coordinate[1])
        if bottom_node not in bottom_node_list:
            data_list = get_shapes(data_list, bottom_node)
            bottom_node_list.append(bottom_node)
    return srs, data_list

if __name__ =='__main__':
    # print('Start Load OSM data:')
    filePath = 'OSM/france.gpkg'
    imagePath = 'image/25-2013-0985-6715-LA93-0M50-E080.tif'
    # imagePath = 'image/33-2012-0375-6450-LA93-0M50-E080.tif'
    # imagePath = 'image/49-2012-0425-6715-LA93-0M50-E080.tif'
    outPath = 'OSM/get_from_image.shp'

    # print('Start Load OSM data:')
    # data_list, srs = readshp(filePath)
    # createKG(data_list, srs, add_node=True)

    print('Start Search KnowLedge Graph:')
    srs, data_list = searchKG(imagePath)
    print(writeshp(outPath, data_list, srs))

    # some test
    # node_list = []
    # rela_list = []
    # node = Node('111', name=1)
    # node_list.append(node)
    # subgraph = Subgraph(node_list, [])
    # g.create(subgraph)
    # node = Node('222', name=2)
    # node_list.append(node)
    # subgraph = Subgraph(node_list, [])
    # g.create(subgraph)

    # rela_list.append(Relationship(node_list[0], '1', node_list[1]))
    # subgraph = Subgraph([], rela_list)
    # g.create(subgraph)

    # rela_list.append(Relationship(node_list[0], '1', node_list[1]))
    # subgraph = Subgraph([], rela_list)
    # g.create(subgraph)

    for imagePath in recursive_glob(r'D:\imgjpg\10(lai20-)rank2','.tif'):
        _,name=os.path.split(imagePath)
        srs, data_list = searchKG(imagePath)
        writetif(name,data_list,srs)
        








