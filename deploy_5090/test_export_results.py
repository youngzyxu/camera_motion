import json,sqlite3,tempfile
from pathlib import Path
from export_results import export_partial,merge_final,digest

def payload(i):
 t={'task_id':str(i),'batch':'chenqi_egomixed_0920'}
 r={'task_id':str(i),'lease_id':'lease'+str(i),'ok':True,'metrics':{'readback_verified':True,'remote':'bos:/test/lease'+str(i),'video_seconds':10}}
 return json.dumps(t),json.dumps(r)
for initial in [0,2]:
 with tempfile.TemporaryDirectory() as d:
  d=Path(d);db=d/'queue.sqlite';c=sqlite3.connect(db)
  c.execute('CREATE TABLE tasks(id TEXT PRIMARY KEY,task TEXT,result TEXT,status TEXT,token TEXT)')
  for i in range(3):
   t,r=payload(i);c.execute('INSERT INTO tasks VALUES (?,?,?,?,?)',(str(i),t,r,'done' if i<initial else 'pending','lease'+str(i)))
  c.commit();meta=export_partial(db,d/'partial.json');assert meta['records']==initial
  meta['bos_readback_sha256_verified']=True
  c.execute("UPDATE tasks SET status='done'");c.commit()
  result=merge_final(db,meta,d/'all.json',3);rows=json.loads((d/'all.json').read_text())
  assert [x['task']['task_id'] for x in rows]==['0','1','2'];assert result['delta_records']==3-initial
  assert digest(d/'all.json')==result['sha256']
  if initial:
   c.execute("UPDATE tasks SET token='replaced' WHERE id='0'");c.commit()
   try:merge_final(db,meta,d/'invalid.json',3)
   except AssertionError:pass
   else:raise AssertionError('Changed accepted lease must be rejected')
  c.close()
print('PASS empty/nonempty partial, delta merge, unique coverage, JSON parse, SHA256 and changed-lease rejection')
