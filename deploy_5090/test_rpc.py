import io,json,os,sys
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError,HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import process_camera_lerobot as p
with patch.dict(os.environ,{'CAMERA_RPC_RETRY':'1'}),patch.object(p.time,'sleep'):
 with patch.object(p.QUEUE_HTTP,'open',side_effect=[URLError('temporary disconnect')]*5+[io.BytesIO(b'{"accepted": true}')]) as call:
  assert p.rpc('http://test','complete',{})['accepted'];assert call.call_count==6
 with patch.object(p.QUEUE_HTTP,'open',side_effect=HTTPError('http://test',400,'invalid request',{},None)) as call:
  try:p.rpc('http://test','complete',{})
  except HTTPError:pass
  else:raise AssertionError('400 should not retry indefinitely')
  assert call.call_count==1
 with patch.object(p.QUEUE_HTTP,'open',side_effect=URLError('down')) as call:
  try:p.rpc('http://test','status',{})
  except URLError:pass
  else:raise AssertionError('Status should fail fast')
  assert call.call_count==1
print('PASS persistent transport recovery beyond three retries, fatal 400, bounded status')
