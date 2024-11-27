from flask import Flask
from flask import request
import pickle
from datetime import datetime
import subprocess
import logging

app = Flask(__name__)


def loadConfig():
    with open('configTS', 'rb') as f:
        ts = pickle.load(f)
        tsVal = {}
        for i in ts.keys():
            tsVal[i] = {'fragmentLen': 0, 'val': [], 'timestamp': []}
        return ts, tsVal

def saveConfig(ts):
    with open('configTS', 'wb') as f:
        pickle.dump(ts, f)
        
def CUDA_TS(ts):
    print(ts)
    subprocess.Popen([r'CUDA_TS\CUDA_TS.exe', f'{ts['name']}_{ts['dimension']}_{ts['fragmentNum']}_TS', f'{ts['fragmentLen']}', f'{ts['subseqLen']}',
    f'mu_sigma\\{ts['name']}_{ts['dimension']}_{ts['fragmentNum']}_MU', f'mu_sigma\\{ts['name']}_{ts['dimension']}_{ts['fragmentNum']}_SIGMA', f'scalar\\{ts['name']}_{ts['dimension']}_{ts['fragmentNum']}_SCALAR'])

@app.route('/')
def configInfo():
    return ts
  
@app.route('/val/', methods=["POST"])
def getVal():
    if tsVal[request.form['name']]['fragmentLen'] < ts[request.form['name']]['fragmentLen']:
        tsVal[request.form['name']]['fragmentLen'] += 1
        tsVal[request.form['name']]['val'].append(request.form['val'])
        tsVal[request.form['name']]['timestamp'].append(datetime.now())
    else:
        with open(f"{request.form['name']}_{ts[request.form['name']]['fragmentNum']}_TS", "w") as file:
            for line in tsVal[request.form['name']]['val']:
                file.write(line + '\n')
                
        tsVal[request.form['name']]['val'] = tsVal[request.form['name']]['val'][ts[request.form['name']]['fragmentLen']-ts[request.form['name']]['subseqLen']:]
        tsVal[request.form['name']]['fragmentLen'] = len(tsVal[request.form['name']]['val'])
        tsVal[request.form['name']]['val'].append(request.form['val'])
        tsVal[request.form['name']]['timestamp'].append(datetime.now())
        
        CUDA_TS(ts[request.form['name']])
        
        ts[request.form['name']]['fragmentNum'] += 1
        saveConfig(ts)
        
        
    return ""

if __name__ == '__main__':
    #log = logging.getLogger('werkzeug') 
    #log.setLevel(logging.ERROR)
    #app.logger.setLevel(logging.CRITICAL)
    ts, tsVal = loadConfig()
    app.run(host='0.0.0.0', port=6060)
