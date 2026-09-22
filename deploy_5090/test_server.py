import concurrent.futures,json,sqlite3,tempfile,time
from pathlib import Path
from server import Queue

def main():
 with tempfile.TemporaryDirectory() as tmp:
  path=Path(tmp)/'q.sqlite';q=Queue(path,lease=.15);q.close()
  db=sqlite3.connect(path)
  db.executemany("INSERT INTO tasks VALUES (?,?,'pending',NULL,0,0,NULL)",[(str(i),json.dumps({'task_id':str(i)})) for i in range(300)])
  db.commit();db.close();q=Queue(path,lease=600)
  with concurrent.futures.ThreadPoolExecutor(max_workers=144) as pool:
   jobs=list(pool.map(lambda _:q.call('claim',{})['task'],range(144)))
   assert len({j['task_id'] for j in jobs})==144
   payloads=[{'task_id':j['task_id'],'lease_id':j['lease_id'],'ok':True} for j in jobs]
   assert all(x['accepted'] for x in pool.map(lambda p:q.call('complete',p),payloads))
  assert q.call('complete',payloads[0])['accepted']
  assert not q.call('heartbeat',{'task_id':jobs[0]['task_id'],'lease_id':'bad'})['accepted']
  assert q.status()['counts']=={'done':144,'pending':156},q.status()
  # Malformed input rolls back its request, leaving subsequent claims viable.
  try:q.call('complete',{})
  except KeyError:pass
  else:raise AssertionError('Malformed request accepted')
  q.close();q=Queue(path,lease=.02);j=q.call('claim',{})['task'];time.sleep(1.05)
  j2=q.call('claim',{})['task'];assert j2['task_id']==j['task_id'] and j2['lease_id']!=j['lease_id']
  assert not q.call('complete',{**j,'ok':True})['accepted']
  q.close();db=sqlite3.connect(path)
  assert db.execute('select attempts from tasks where id=?',(j['task_id'],)).fetchone()[0]==1
  assert db.execute('pragma integrity_check').fetchone()[0]=='ok';db.close()
 print('PASS: 144 concurrent clients, idempotence, stale rejection, rollback, restart, lease recovery and SQLite integrity')
if __name__=='__main__':main()
