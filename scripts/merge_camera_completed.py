"""Export an accepted-result index at a fixed cutoff without stopping production."""
import concurrent.futures,datetime,gzip,hashlib,json,pathlib,sqlite3,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/hand2_20260915'
NODES=['pro6000-1','pro6000-2','pro6000-7']
def main():
 cutoff=time.time();stamp=datetime.datetime.fromtimestamp(cutoff,datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=ROOT/'results'/('merged_camera_'+stamp);out.mkdir()
 meta={'snapshot_utc':datetime.datetime.fromtimestamp(cutoff,datetime.timezone.utc).isoformat(),'cutoff_unix_s':cutoff,'kind':'accepted_result_index','contains_pose_arrays':False,'nodes':{},'config':None}
 (out/'snapshot.json').write_text(json.dumps(meta,indent=2));print('OUTPUT',out,flush=True)
 def export(node):
  c=sqlite3.connect('file:'+str(RUN/node/'queue.sqlite')+'?mode=ro',uri=True,timeout=120)
  count=0;seconds=0.;p=out/(node+'.jsonl.gz')
  with gzip.open(p,'wt',compresslevel=3) as f:
   for task_id,raw in c.execute("SELECT id,result FROM tasks WHERE status='done'"):
    r=json.loads(raw)
    if r['accepted_at']>cutoff:continue
    m=r['metrics'];t=m['task'];assert r['ok'] and m['readback_verified'] and t['hand_quality']==2
    assert task_id==t['task_id'] and r['lease_id']==t['lease_id']
    remote=m['remote'];assert remote.endswith('/'+r['lease_id'])
    record={'task_id':task_id,'episode_index':t['episode_index'],'episode':t['episode'],'chunk':t['chunk'],'camera':t['camera'],'relative_video':t['relative_video'],'video':m['video'],'hand_quality':2,'lease_id':r['lease_id'],'remote':remote,'camera_npz':remote+'/camera.npz','camera_json':remote+'/camera.json','success_json':remote+'/SUCCESS.json','config':m['config'],'video_seconds':m['video_seconds'],'sampled_frames':m['sampled_frames'],'pose_convention':m['pose_convention'],'readback_verified':True,'accepted_at':r['accepted_at'],'node':node}
    f.write(json.dumps(record,separators=(',',':'))+'\n');count+=1;seconds+=m['video_seconds']
    if count%50000==0:print(node,count,flush=True)
  c.close();print('EXPORTED',node,count,flush=True);return node,{'count':count,'video_hours':seconds/3600}
 with concurrent.futures.ThreadPoolExecutor(3) as pool:
  meta['nodes']=dict(pool.map(export,NODES))
 seen=set();count=0;dst=out/'camera_pose_completed.jsonl.gz'
 with gzip.open(dst,'wt',compresslevel=3) as f:
  for node in NODES:
   with gzip.open(out/(node+'.jsonl.gz'),'rt') as src:
    for line in src:
     r=json.loads(line);assert r['task_id'] not in seen,('duplicate',r['task_id']);seen.add(r['task_id']);f.write(line);count+=1
     if meta['config'] is None:meta['config']=r['config']
     assert r['config']==meta['config']
 assert count==sum(x['count'] for x in meta['nodes'].values())
 sha=hashlib.sha256()
 with dst.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):sha.update(block)
 meta.update(count=count,video_hours=sum(x['video_hours'] for x in meta['nodes'].values()),file=dst.name,bytes=dst.stat().st_size,sha256=sha.hexdigest(),unique_task_ids=True)
 (out/'snapshot.json').write_text(json.dumps(meta,indent=2));(out/'SHA256SUMS').write_text(sha.hexdigest()+'  '+dst.name+'\n')
 (out/'README.md').write_text('# Camera pose 已完成结果合并快照\n\n这是三台生产队列已接受成功结果的 JSONL gzip 索引，不内嵌 pose 数组。每行包含 episode、原视频、配置、有效 lease_id，以及 BOS 上 camera.npz / camera.json / SUCCESS.json 路径。NPZ 内包含 c2w、intrinsics、frame_indices、timestamps_s 等数组。\n\n只包含 accepted_at 不晚于 snapshot.json 截止时间的结果，排除诊断输出、失败、在途及未被队列接受的重试目录。每个 task_id 唯一。生产任务继续运行；这是快照，不是最终全集。各条 readback_verified 来自生产上传时的 SHA256 回读校验，本次没有重新下载所有逐视频文件。\n')
 print('DONE',json.dumps(meta),flush=True)
if __name__=='__main__':main()
