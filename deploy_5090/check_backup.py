"""Validate a pinned online snapshot while production writers remain active."""
import sqlite3,json,time
from pathlib import Path
p=Path('/tmp/camera_motion_5090/chenqi_egomixed_0920_5090/queue.sqlite')
out=p.with_name('backup_validation.sqlite');t=time.monotonic()
s=sqlite3.connect(p,timeout=30);d=sqlite3.connect(out)
s.execute('BEGIN');s.execute('SELECT count(*) FROM sqlite_master').fetchone()
s.backup(d,pages=4096,sleep=.01);s.close()
counts=dict(d.execute('SELECT status,count(*) FROM tasks GROUP BY status'))
assert sum(counts.values())==1169951 and counts.get('done',0)>=443805
assert d.execute('PRAGMA quick_check').fetchone()[0]=='ok';d.close();out.unlink()
r={'seconds':time.monotonic()-t,'counts':counts,'quick_check':'ok','pinned_online_snapshot':True}
Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/xuzhiyong/explore_project/camera_motion/runs/chenqi_egomixed_0920_5090/backup_validation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
