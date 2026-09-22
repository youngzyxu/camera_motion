"""Read back one accepted result per active GPU and validate endpoints/pose finiteness."""
import argparse,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from process_camera_lerobot import transfer
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'runs/chenqi_egomixed_0920_5090');a=p.parse_args()
saved=a.run/'gpu_output_audit.json'
checks=json.loads(saved.read_text()) if saved.exists() else []
verified={(x['node'],x['gpu']) for x in checks}
for host in json.loads((a.run/'topology.json').read_text())['client_nodes']:
 for gpu in range(8):
  if (host,gpu) in verified:continue
  files=sorted((a.run/host/f'gpu{gpu}').glob('*/completed.jsonl'))
  if not files:continue
  line=next((s for s in files[-1].read_text().splitlines() if s.strip()),None)
  if not line:continue
  row=json.loads(line)
  with tempfile.TemporaryDirectory() as tmp:
   dst=Path(tmp)/'camera.npz';transfer(row['remote']+'/camera.npz',dst)
   with np.load(dst) as z:
    assert z['frame_indices'][-1]==row['source_frames']-1
    assert np.isfinite(z['c2w']).all() and np.isfinite(z['intrinsics']).all()
    assert len(z['c2w'])==row['sampled_frames']
  assert row['readback_verified'] and row['config']['window']==240 and row['config']['overlap']==36
  checks.append({'node':host,'gpu':gpu,'task_id':row['task']['task_id'],'remote':row['remote'],'last_frame_verified':True})
(a.run/'gpu_output_audit.json').write_text(json.dumps(checks,indent=2));print('VALIDATED',len(checks),'GPU outputs')
