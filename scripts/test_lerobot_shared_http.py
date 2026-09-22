"""Two client identities share the real HTTP server; a vanished client is reclaimed."""
import concurrent.futures,json,pathlib,socket,subprocess,sys,tempfile,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from process_camera_lerobot import rpc
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 with tempfile.TemporaryDirectory() as tmp:
  d=pathlib.Path(tmp);manifest=d/'tasks.jsonl';manifest.write_text(''.join(json.dumps({'task_id':str(i),'video':'test'})+'\n' for i in range(12)))
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  url=f'http://127.0.0.1:{port}'
  def start():
   p=subprocess.Popen([sys.executable,'-u',str(ROOT/'process_camera_lerobot.py'),'server','--manifest',str(manifest),'--db',str(d/'queue.sqlite'),'--host','127.0.0.1','--port',str(port),'--lease-seconds','3'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   assert 'SERVER_READY' in p.stdout.readline();return p
  p=start()
  try:
   with concurrent.futures.ThreadPoolExecutor(8) as pool:
    tasks=list(pool.map(lambda i:rpc(url,'claim',{'client_id':f'node-{i%2}:gpu-{i}'})['task'],range(12)))
   assert len({t['task_id'] for t in tasks})==12
   assert {t['worker_id'].split(':')[0] for t in tasks}=={'node-0','node-1'}
   for t in tasks[1:]:assert rpc(url,'complete',{**t,'ok':True})['accepted']
   lost=tasks[0];time.sleep(3.2)
   rescued=rpc(url,'claim',{'client_id':'remaining-node'})['task']
   assert rescued['task_id']==lost['task_id'] and rescued['lease_id']!=lost['lease_id']
   assert rescued['worker_id']=='remaining-node'
   assert not rpc(url,'complete',{**lost,'ok':True})['accepted']
   assert rpc(url,'complete',{**rescued,'ok':True})['accepted']
   assert rpc(url,'status',{})['counts']=={'done':12}
  finally:p.terminate();p.wait(timeout=10)
  p=start()
  try:
   assert rpc(url,'status',{})['counts']=={'done':12}
   assert rpc(url,'claim',{'client_id':'new-node'})=={'task':None,'exhausted':True}
  finally:p.terminate();p.wait(timeout=10)
 print('PASS real HTTP, concurrent clients, vanished-client reassignment, stale lease rejection, restart durability')
if __name__=='__main__':main()
