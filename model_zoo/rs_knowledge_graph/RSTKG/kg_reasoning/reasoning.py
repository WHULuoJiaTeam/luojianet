import encodings
import numpy as np
from py2neo import Graph, Node, Relationship, NodeMatcher, Subgraph
import time as Time
from osgeo import gdal
import csv

import rasterio
from PIL import Image
import os

clf_name=['','建筑','基础设施','工矿用地','城市绿地','耕地','园地','牧场','','','森林','灌木','裸地','湿地','水体']


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


g = Graph('http://localhost:7474/', auth=('neo4j', '123456'))

# 构建推理规则知识图谱
def create_rules(filePath):
    data = []
    with open(filePath, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        data = list(reader)
        f.close()

    print('begin to load data')
    for i in range(len(data)):
        # 用于匹配当前图中已经存在的节点
        matcher = NodeMatcher(g)
        # 匹配三元组主谓宾中的主语
        nodelist = list(matcher.match(data[i][1], name=data[i][0], source_class=data[i][2]))
        # 如果nodelist长度大于0，说明当前图中已经有相同name属性的主语节点
        if len(nodelist) > 0:
            print(nodelist)
            #获取这个已经存在的主语节点
            znode = nodelist[0]
            #继续判断宾语节点是否存在于当前图中
            matcher = NodeMatcher(g)
            # nodelist = list(matcher.match('宾语',name=data[i][1]))
            nodelist = list(matcher.match(data[i][5], name=data[i][4], source_class=data[i][6]))
            # 如果宾语节点也存在于当前图中，直接获取这个节点
            if len(nodelist) > 0:
                print(nodelist)
                bnode = nodelist[0]
                # 建立它们之间新的关系，避免重复建立节点
                z2b = Relationship(znode, data[i][3], bnode)
                g.create(z2b)
            # 新建一个宾语节点
            else:
                bnode = Node(data[i][5], name=data[i][4], source_class=data[i][6])
                g.create(bnode)
                z2b = Relationship(znode, data[i][3], bnode)
                g.create(z2b)
        # 新建一个主语节点
        else:
            znode = Node(data[i][1], name=data[i][0], source_class=data[i][2])
            g.create(znode)
            matcher = NodeMatcher(g)
            nodelist = list(matcher.match(data[i][5], name=data[i][4], source_class=data[i][6]))
            # 判断宾语节点是否已经存在，和上面的类似
            if len(nodelist) > 0:
                print(nodelist)
                bnode = nodelist[0]
                z2b = Relationship(znode, data[i][3], bnode)
                g.create(z2b)
            else:
                bnode = Node(data[i][5], name=data[i][4], source_class=data[i][6])
                g.create(bnode)
                z2b = Relationship(znode, data[i][3], bnode)
                g.create(z2b)
    return

# 进行规则推理
def kg_reasoning(clf_result, get_conditions):
    type = 'match (n{name:"' + clf_result + '"})-[r:修改为]->(q) '
    results = []
    if get_conditions == []:
        get_conditions = ['']
        
    for (i, condition) in enumerate(get_conditions):
        condi = 'with q match (p' + str(i) + '{name:"' + condition + '", source_class:"' + clf_result + '"})-[r:条件]->(q) '
        # print(condi)
        cyper = type + condi + 'return q'
        print(cyper)
        tmp=g.run(cyper).data()
        if tmp!=[]:
            results.append(tmp[0])

    print(results)

    if len(results) > 0:
        for result in results:
            get_necessary_conditions = 'match (p:必要条件{source_class:"' + clf_result + '"})-[r:条件]->(q{name:"' + result['q']['name'] + '"}) return p'
            necessary = g.run(get_necessary_conditions).data()
            flag = [True for node in necessary if node['p']['name'] not in get_conditions]
            if not flag:
                print('可选的修改方案：修改为"' + result['q']['name'] + '"满足设置条件。')
                totype=clf_name.index(result['q']['name'])
                print(totype)
                return totype
            else:
                print('可选的修改方案：修改为"' + result['q']['name'] + '"不满足必要条件。')
    else:
        print('已知条件下无已设置的修改方案。')
    return

if __name__ == '__main__':
    # filePath = 'rules_kg4.csv'
    # create_rules(filePath)
    
    # print(clf_name[1])
    # print(clf_name[14])
    # clf_result=clf_name[1]

    # clf_result = '森林'
    # clf_result = '水体'

    # # set_conditions = ['错误分类']
    # set_conditions = ['与建筑、工矿用地、城市绿地相邻长度占邻接边总长度一半以上']
    # # set_conditions = ['错误分类', '图幅边缘', '被包围']
    # set_conditions = ['被其包围']
    # kg_reasoning(clf_result, set_conditions)

    # clf_result = '裸地'
    # set_conditions=['与建筑、工矿用地、城市绿地相邻长度占邻接边总长度一半以上','与其相邻']
    # totype=kg_reasoning(clf_result, set_conditions)
    # print(totype)



    for predname in recursive_glob(r"D:\imgjpg\test",'.tif'):
        if not os.path.exists(predname.replace('test','20220505kg2')):
            a,b=os.path.split(predname)
            print(b)

            pred=rasterio.open(predname).read()
            pred=np.squeeze(pred)

            allid=np.load(os.path.join(r"D:\imgjpg\test8liantong",b.replace('.tif',"_allid.npy")))
            allid=allid.astype(np.uint8)
            allidshp=np.shape(allid)

            alllbl=np.load(os.path.join(r"D:\imgjpg\test8liantong",b.replace('.tif',".tif.npy")))

            unitname=os.path.join(r"D:\imgjpg\test8liantong",b)
            unit=rasterio.open(unitname).read()
            unit=np.squeeze(unit)

            lencntname=os.path.join(r"D:\imgjpg\test8liantong",b.replace('.tif',"_cnt.tif"))
            lencnt=rasterio.open(lencntname).read()
            lencnt=np.squeeze(lencnt)

            for i in range(len(allid)):
                set_conditions=[]

                clf_result=clf_name[allid[i]]
                set0=set(allid[np.where(alllbl[i]==1)[0]])

                if((set([1])<=set0)|(set([2])<=set0)|(set([3])<=set0)|(set([4])<=set0)):
                    alllen=np.sum(lencnt[i])
                    tmp=0
                    for j in np.where(alllbl[i]==1)[0]:
                        if allid[j] in [1,3,4]:
                            tmp=tmp+lencnt[i][j]
                    urbanrate=tmp/alllen
                    print(urbanrate)
                    if urbanrate>0.5:
                        set_conditions.append( '与建筑、工矿用地、城市绿地相邻长度占邻接边总长度一半以上')
                if(set0>=set([3])):
                    set_conditions.append( '与其相邻')
                if(set0<=set([14])):
                    set_conditions.append( '被水体包围')
                if(set0<=set([1])):
                    set_conditions.append('被建筑包围')
                if(set0<=set([2])):
                    set_conditions.append('被基础设施包围')
                if(set0<=set([3])):
                    set_conditions.append('被工矿用地包围')

                print(set_conditions)

                totype=kg_reasoning(clf_result, set_conditions)
                if(totype is not None):
                    pred[np.where(unit==i)]=totype
                    totype=None
                

            datatype=gdal.GDT_Byte
            driver = gdal.GetDriverByName('GTiff')  # 数据类型必须有，因为要计算需要多大内存空间
            dataset = driver.Create(predname.replace('test','20220505kg2'), 10000, 10000, 1, datatype,options=["TILED=YES","COMPRESS=DEFLATE"])
            dataset.GetRasterBand(1).WriteArray(pred)  # 写入数组数据
            del dataset

            seg_img = Image.fromarray(pred).convert('P')
            seg_img.putpalette(np.array(palette, dtype=np.uint8))
            seg_img.save(predname.replace('test','20220505kg2').replace('tif','png'))








