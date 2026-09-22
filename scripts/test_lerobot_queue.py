"""Exercise lease expiry, stale writes, restart durability and concurrent claims."""
import json,sys,tempfile,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from process_camera_lerobot import Queue

def main():
 with tempfile.TemporaryDirectory() as d:
  d=Path(d);m=d/'tasks.jsonl';m.write_text(''.join(json.dumps({'task_id':str(i),'video':'test'})+'\n' for i in range(20)))
  q=Queue(d/'queue.db',m,lease=10,max_attempts=2)
  with ThreadPoolExecutor(8) as pool:ts=list(pool.map(lambda _:q.call('claim',{})['task'],range(20)))
  assert len({t['task_id'] for t in ts})==20
  assert q.call('claim',{})=={'task':None,'exhausted':False}
  t=ts[0];assert q.call('complete',{**t,'ok':True})['accepted'];assert q.call('complete',{**t,'ok':True})['accepted']
  q.db.execute('UPDATE tasks SET deadline=0 WHERE id=?',(ts[1]['task_id'],));q.db.commit()
  new=q.call('claim',{})['task'];assert new['task_id']==ts[1]['task_id'] and new['lease_id']!=ts[1]['lease_id']
  assert not q.call('complete',{**ts[1],'ok':True})['accepted']
  assert q.call('heartbeat',new)['accepted'];assert q.call('complete',{**new,'ok':False})['accepted']
  assert q.call('status',{})['counts']['failed']==1
  q.db.close();restored=Queue(d/'queue.db',m)
  assert restored.call('status',{})['counts']['done']==1
  assert not restored.call('complete',{**t,'lease_id':'bad','ok':True})['accepted']
 print('PASS concurrent claims, expiry/retry, stale rejection, heartbeat, idempotent ack, restart persistence')
if __name__=='__main__':main()
