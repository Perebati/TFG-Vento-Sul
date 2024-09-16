import numpy as np
from scipy.io import loadmat
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from model import dataSetModel

Data = dataSetModel.DataSet

dados = loadmat('P1_LIDAR_matrix.mat') 
wspeed = dados['L']['wspeed'][0][0]
mtime = dados['L']['mtime'][0][0]
temp = dados['L']['temp'][0][0]
vspeed = dados['L']['vertspeed'][0][0]
wdir = dados['L']['wdir'][0][0]

mtime = mtime.reshape(-1)

dt_index = pd.to_datetime(mtime - 719529, unit='D', origin='unix')

engine = create_engine('postgresql://root:root@localhost:5432/DataSet-PerfilVento')

Session = sessionmaker(bind=engine)
session = Session()
dataSetModel.Base.metadata.create_all(engine)

for i in range(np.size(wspeed, 0)):
    obj = Data(
        dt_index[i],
        dt_index[i].year,
        dt_index[i].month,
        dt_index[i].day,
        dt_index[i].hour,
        dt_index[i].minute,
        wspeed[i][0],
        wspeed[i][1],
        wspeed[i][2],
        wspeed[i][3], 
        wspeed[i][4],
        wspeed[i][5],
        wspeed[i][6],
        wspeed[i][7],
        wspeed[i][8],
        wspeed[i][9],
        wspeed[i][10],
        wspeed[i][11],
        wspeed[i][12],
        wspeed[i][13],
        wspeed[i][14],
        wspeed[i][15],
        wspeed[i][16],
        wspeed[i][17],
        wspeed[i][18],
        wspeed[i][19],
        vspeed[i][0],
        vspeed[i][1],
        vspeed[i][2],
        vspeed[i][3],
        vspeed[i][4],
        vspeed[i][5],
        vspeed[i][6],
        vspeed[i][7],
        vspeed[i][8],
        vspeed[i][9],
        vspeed[i][10],
        vspeed[i][11],
        vspeed[i][12],
        vspeed[i][13],
        vspeed[i][14],
        vspeed[i][15],
        vspeed[i][16],
        vspeed[i][17],
        vspeed[i][18],
        vspeed[i][19],
        wdir[i][0],
        wdir[i][1],
        wdir[i][2],
        wdir[i][3],
        wdir[i][4],
        wdir[i][5],
        wdir[i][6],
        wdir[i][7],
        wdir[i][8],
        wdir[i][9],
        wdir[i][10],
        wdir[i][11],
        wdir[i][12],
        wdir[i][13],
        wdir[i][14],
        wdir[i][15],
        wdir[i][16],
        wdir[i][17],
        wdir[i][18],
        wdir[i][19],
        (wspeed[i][1] -  wspeed[i][0]) / 10,
        (wspeed[i][2] -  wspeed[i][1]) / 10,
        (wspeed[i][3] -  wspeed[i][2]) / 10,
        (wspeed[i][4] -  wspeed[i][3]) / 10,
        (wspeed[i][5] -  wspeed[i][4]) / 10,
        (wspeed[i][6] -  wspeed[i][5]) / 10,
        (wspeed[i][7] -  wspeed[i][6]) / 10,
        (wspeed[i][8] -  wspeed[i][7]) / 10,
        (wspeed[i][9] -  wspeed[i][8]) / 10,
        (wspeed[i][10] -  wspeed[i][9]) / 10,
        (wspeed[i][11] -  wspeed[i][10]) / 10,
        (wspeed[i][12] -  wspeed[i][11]) / 10,
        (wspeed[i][13] -  wspeed[i][12]) / 10,
        (wspeed[i][14] -  wspeed[i][13]) / 10,
        (wspeed[i][15] -  wspeed[i][14]) / 10,
        (wspeed[i][16] -  wspeed[i][15]) / 10,
        (wspeed[i][17] -  wspeed[i][16]) / 20,
        (wspeed[i][18] -  wspeed[i][17]) / 20,
        (wspeed[i][19] -  wspeed[i][18]) / 20
    )
    session.add(obj)
    session.commit()


