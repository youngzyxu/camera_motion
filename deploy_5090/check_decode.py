import json,sys,hashlib
from pathlib import Path
import cv2
p=Path(sys.argv[1]);video=json.loads(p.read_text().splitlines()[0])['video']
def digest(threads):
 cap=cv2.VideoCapture(video) if threads is None else cv2.VideoCapture(video,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,threads])
 h=hashlib.sha256();n=0
 while True:
  ok,frame=cap.read()
  if not ok:break
  h.update(frame.tobytes());n+=1
 cap.release();return n,h.hexdigest()
a=digest(None);b=digest(1);assert a==b and a[0]>0,(a,b)
print(json.dumps({'video':video,'frames':a[0],'pixel_sha256':a[1],'default_vs_one_thread_identical':True}))
