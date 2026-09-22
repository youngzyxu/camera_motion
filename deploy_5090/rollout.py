"""Gracefully roll node supervisors to current code, bounded three nodes at a time."""
import concurrent.futures,json,time,sys
from manage import RUN,ROOT,PY,ssh,detached
hosts=json.loads((RUN/'topology.json').read_text())['client_nodes']
def update(host):
 folder=RUN/host;pid=int((folder/'node.pid').read_text());(folder/'STOP').touch()
 for _ in range(600):
  state=ssh(host,f'ps -p {pid} -o stat= || true')
  if not state or state.startswith('Z'):break
  time.sleep(2)
 else:raise RuntimeError(host+' did not finish drain; kept stopped')
 (folder/'STOP').unlink()
 pid=detached(host,'node',[PY,'-u',str(ROOT/'deploy_5090/node.py'),'--node',host,'--run',str(RUN)])
 print('ROLLED',host,pid,flush=True)
 return host
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 list(pool.map(update,hosts))
print('ROLLOUT_COMPLETE',flush=True)
