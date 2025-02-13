from datetime import datetime
import clickhouse_connect
import random
import numpy as np

import ctypes

def CUDA_full(name, subseqLen, fragmentLen, fragmentNum, ts, dimension):

    lib = ctypes.CDLL("C:\\Project\\Preprocessing\\CUDA_TS.dll")

    TS = (ctypes.c_double * (fragmentLen + subseqLen - 1))()

    cSubseqLen = ctypes.c_int(subseqLen)

    mu = (ctypes.c_double * fragmentLen)()
    sigma = (ctypes.c_double * fragmentLen)()
    mp = (ctypes.c_double * fragmentLen)()
    mpIndex = (ctypes.c_int * fragmentLen)()

    ctypes.memmove(TS, ts[:,:1,].transpose()[0].ctypes.data, ts[:,:1,].transpose()[0].nbytes)

    lib.cudaRun_All(TS, cSubseqLen, mu, sigma, mp, mpIndex)

    ts = np.ctypeslib.as_array(TS)

    mu = np.ctypeslib.as_array(mu).reshape(fragmentLen, 1)
    sigma = np.ctypeslib.as_array(sigma).reshape(fragmentLen, 1)
    mp = np.ctypeslib.as_array(mp).reshape(fragmentLen, 1)
    mpIndex = np.ctypeslib.as_array(mpIndex).reshape(fragmentLen, 1)

    column_names = [f"mu_{subseqLen}_{dimension}", f"sigma_{subseqLen}_{dimension}", "fragment","num"]

    send = np.hstack((mu, sigma, np.full(fragmentLen,fragmentNum).reshape(fragmentLen, 1), np.arange(1, fragmentLen + 1).reshape(fragmentLen, 1)))

    client.insert(f'default.FT_{name}', send, column_names=column_names)

    column_names = [f"NNdistance_{subseqLen}_{dimension}", f"NNindex_{subseqLen}_{dimension}", f"NNindexFragment_{subseqLen}_{dimension}", "fragment","num"]
  
    send = np.hstack((mp, mpIndex, np.full(fragmentLen,fragmentNum).reshape(fragmentLen, 1), np.full(fragmentLen,fragmentNum).reshape(fragmentLen, 1), np.arange(1, fragmentLen + 1).reshape(fragmentLen, 1)))

    client.insert(f'default.MP_{name}', send, column_names=column_names)
    
def sendDBTS(name, subseqLen, ts, dimension):
    
    column_names = [f"val_{subseqLen}_{dimension}","timeStamp", "fragment","num"]
    client.insert(f'default.TS_{name}', ts, column_names=column_names)

    # client.command(f"ALTER TABLE default.METADATA_TS UPDATE name={name}, dimension={dimension}, subseqLen={ts[name]['subseqLen']}, fragmentNum={ts[name]['fragmentNum']}, fragmentComp={ts[name]['fragmentComp']}")

def getVal(name, subseqLen, fragmentLen, fragmentComp, fragmentNum, tsNum, ts, tsLen, dimension):

    test=open('file.txt', 'a')

    for i in range(tsNum):
        if tsLen < fragmentLen + subseqLen - 1:
            num = round(random.uniform(1,100), 8)
            ts[tsLen][0] = num
            ts[tsLen][1] = datetime.timestamp(datetime.now())
            ts[tsLen][2] = fragmentNum

            test.write(f'{num}, ')
            
            tsLen += 1
        else:
            sendDBTS(name, subseqLen, ts, dimension)
            CUDA_full(name, subseqLen, fragmentLen, fragmentNum, ts, dimension)

            fragmentNum += 1
            tsLen = subseqLen - 1
            ts[:,:1,][:subseqLen-1] = ts[:,:1,][fragmentLen:]
            ts[:,2:3,][:subseqLen-1] += 1
    
    test.close()

if __name__ == '__main__':
    client = clickhouse_connect.get_client(host='localhost', port='8123', user='default', password= '')

    name = "THERMALSENSOR"
    subseqLen = 256
    fragmentLen = 30000
    tsLen = 0
    fragmentComp = 0
    fragmentNum = 1
    tsNum = 3000260
    ts = np.hstack((np.zeros((3, fragmentLen + subseqLen - 1)).transpose(), np.array([np.arange(1, fragmentLen + subseqLen)]).transpose()))
    dimension = 1
    getVal(name, subseqLen, fragmentLen, fragmentComp, fragmentNum, tsNum, ts, tsLen, dimension)

    print(type(datetime.now()))