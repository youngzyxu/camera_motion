"""Deploy only config-listed 5090 hosts; never starts old Pro6000 workers."""
import argparse,concurrent.futures,json,re,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/chenqi_egomixed_0920_5090'
PY='/mnt/pfs/pfs-yc2F4O/miniconda3/envs/camera_motion/bin/python'

def ssh(host,cmd,timeout=60):
 p=subprocess.run(['ssh','-F','/root/.ssh/config','-o','BatchMode=yes','-o','ConnectTimeout=10',host,cmd],text=True,capture_output=True,timeout=timeout)
 if p.returncode:raise RuntimeError(host+': '+p.stderr[-2000:]+p.stdout[-2000:])
 return p.stdout.strip()

def detached(host,name,args):
 folder=RUN/host;folder.mkdir(exist_ok=True)
 code="""import pathlib,subprocess,os
folder=pathlib.Path(%r);pidfile=folder/%r
if pidfile.exists():
 pid=int(pidfile.read_text());p=pathlib.Path('/proc')/str(pid)/'cmdline'
 if p.exists() and %r.encode() in p.read_bytes():raise SystemExit('Already running')
log=(folder/%r).open('a')
p=subprocess.Popen(%r,cwd=%r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(p.pid));print(p.pid)
"""%(str(folder),name+'.pid',str(ROOT/'deploy_5090'),name+'.log',args,str(ROOT))
 return ssh(host,shlex.join([PY,'-c',code]))

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['init','server','clients','status','stop']);p.add_argument('--hosts');p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--seconds',type=int,default=0);a=p.parse_args()
 if a.action=='init':
  hosts=re.findall(r'^Host (5090-\d+)\b',Path('/root/.ssh/config').read_text(),re.M)
  assert hosts and len(hosts)==len(set(hosts));RUN.mkdir(exist_ok=True)
  if (RUN/'topology.json').exists():raise SystemExit('Run already initialized')
  inventory=json.loads((ROOT/'runtime/5090/inventory.json').read_text())
  ip=inventory[hosts[0]]['stdout'].split()[0]
  top={'server_node':hosts[0],'client_nodes':hosts,'bind_host':ip,'port':18818,'server_url':f'http://{ip}:18818','queue_db':'/tmp/camera_motion_5090/chenqi_egomixed_0920_5090/queue.sqlite','checkpoint':str(RUN/'global/checkpoint.sqlite'),'restore':str(ROOT/'runs/chenqi_egomixed_0920/global/checkpoint.sqlite')}
  (RUN/'global').mkdir();(RUN/'topology.json').write_text(json.dumps(top,indent=2))
  (RUN/'inventory.json').write_bytes((ROOT/'runs/chenqi_egomixed_0920/inventory.json').read_bytes())
  (RUN/'resume_origin.json').write_bytes((ROOT/'runs/chenqi_egomixed_0920/stop_request.json').read_bytes())
  print(json.dumps(top,indent=2));return
 top=json.loads((RUN/'topology.json').read_text());hosts=a.hosts.split(',') if a.hosts else top['client_nodes']
 assert set(hosts)<=set(top['client_nodes'])
 if a.action=='server':
  print(detached(top['server_node'],'server',[PY,'-u',str(ROOT/'deploy_5090/server.py'),'--db',top['queue_db'],'--restore',top['restore'],'--checkpoint',top['checkpoint'],'--host',top['bind_host'],'--port',str(top['port'])]));return
 if a.action=='clients':
  if (RUN/'STOP').exists():raise SystemExit('Run stopped; clear STOP explicitly before resuming')
  def launch(h):return h,detached(h,'node',[PY,'-u',str(ROOT/'deploy_5090/node.py'),'--node',h,'--run',str(RUN),'--gpus',a.gpus,'--seconds',str(a.seconds)])
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
   for h,result in pool.map(launch,hosts):print(h,result,flush=True)
 elif a.action=='stop':
  for h in hosts:(RUN/h/'STOP').touch()
  print('Client drain requested; keep server online until running=0, then SIGTERM server for final checkpoint.')
 else:
  sys.path.insert(0,str(ROOT));from process_camera_lerobot import rpc
  print(json.dumps(rpc(top['server_url'],'status',{}),indent=2))
if __name__=='__main__':main()
