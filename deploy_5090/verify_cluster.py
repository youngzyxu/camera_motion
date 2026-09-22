"""Read-only deployment audit; prints only camera-specific process flags, no credentials."""
import concurrent.futures,json,shlex,time
from manage import RUN,PY,ssh
hosts=json.loads((RUN/'topology.json').read_text())['client_nodes']
def verify(host):
 code="""import json,pathlib,subprocess
state=json.loads(pathlib.Path(%r).read_text());workers=[]
for gpu,w in state['workers'].items():
 p=pathlib.Path('/proc')/str(w['pid']);live=p.exists()
 flags={}
 if live:
  env=p.joinpath('environ').read_bytes().split(b'\\0')
  flags={k:next((v.split(b'=',1)[1].decode() for v in env if v.startswith(k.encode()+b'=')),None) for k in ['CAMERA_RPC_RETRY','CAMERA_PREFETCH_LOCAL','CAMERA_DECODE_THREADS','CAMERA_UPLOAD_WORKERS']}
 workers.append({'gpu':gpu,'pid':w['pid'],'live':live,'flags':flags})
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
print(json.dumps({'workers':workers,'gpu_csv':gpu}))
"""%str(RUN/host/'state.json')
 return host,json.loads(ssh(host,shlex.join([PY,'-c',code])))
with concurrent.futures.ThreadPoolExecutor(max_workers=18) as pool:data=dict(pool.map(verify,hosts))
report={'unix':time.time(),'nodes':data,'all_ready':all(len(x['workers'])==8 and all(w['live'] and w['flags'].get('CAMERA_RPC_RETRY')=='1' for w in x['workers']) for x in data.values())}
(RUN/'cluster_validation.json').write_text(json.dumps(report,indent=2))
print('nodes',len(data),'workers',sum(len(x['workers']) for x in data.values()),'all_ready',report['all_ready'])
assert report['all_ready']
