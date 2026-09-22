"""Single-writer batched SQLite dispatcher, compatible with existing task/result schema.
Local durable commits precede replies. O(1) status; isolated PFS snapshot thread.
"""
import argparse, collections, concurrent.futures, json, os, queue, shutil, signal
import sqlite3, threading, time, uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Queue:
    def __init__(self, path, lease=600, max_attempts=3):
        self.path = str(path)
        self.lease, self.max_attempts = lease, max_attempts
        self.inbox = queue.Queue(2048)
        self.metrics_lock = threading.Lock()
        self.latencies = collections.defaultdict(lambda: collections.deque(maxlen=2000))
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.writer, daemon=True)
        self.thread.start()
        self.ready.wait()
        if self.error: raise self.error

    def status(self):
        with self.metrics_lock:
            return {'counts': self.published.copy(), 'rpc_ms': {
                k: {'n': len(v), 'mean': sum(v)/len(v), 'p95': sorted(v)[int((len(v)-1)*.95)]}
                for k,v in self.latencies.items() if v}, 'writer_backlog': self.inbox.qsize()}

    def call(self, op, p):
        if op == 'health': return {'ok': self.error is None}
        if op == 'status': return self.status()
        if op not in ('claim','heartbeat','complete'): raise ValueError(op)
        if self.error: raise RuntimeError(str(self.error))
        start = time.monotonic()
        f = concurrent.futures.Future()
        self.inbox.put((op,p,f),timeout=5)
        result = f.result(timeout=120)
        with self.metrics_lock: self.latencies[op].append((time.monotonic()-start)*1000)
        return result

    def change(self, old, new, n=1):
        self.counts[old] = self.counts.get(old,0)-n
        self.counts[new] = self.counts.get(new,0)+n

    def expire(self, now):
        n = self.db.execute("SELECT count(*) FROM tasks WHERE status='running' AND deadline<?",(now,)).fetchone()[0]
        if n:
            self.db.execute("UPDATE tasks SET status='pending',token=NULL,deadline=0,attempts=max(0,attempts-1) WHERE status='running' AND deadline<?",(now,))
            self.change('running','pending',n)

    def apply(self, op, p, now):
        if op == 'claim':
            r = self.db.execute("SELECT id,task FROM tasks INDEXED BY tasks_pending_id WHERE status='pending' ORDER BY id LIMIT 1").fetchone()
            if not r: return {'task': None, 'exhausted': not self.counts.get('running',0)}
            token = uuid.uuid4().hex
            task = json.loads(r[1])
            self.db.execute("UPDATE tasks SET status='running',token=?,deadline=?,attempts=attempts+1 WHERE id=?",(token,now+self.lease,r[0]))
            self.change('pending','running')
            return {'task': {**task,'lease_id': token,'worker_id': p.get('client_id','unknown')}}
        r = self.db.execute('SELECT status,token,attempts FROM tasks WHERE id=?',(p['task_id'],)).fetchone()
        if not r or r[1] != p['lease_id']: return {'accepted': False,'reason':'stale lease'}
        if op == 'complete' and r[0] == 'done': return {'accepted': True}
        if r[0] != 'running': return {'accepted': False,'reason':'not running'}
        if op == 'heartbeat':
            self.db.execute('UPDATE tasks SET deadline=? WHERE id=?',(now+self.lease,p['task_id']))
        else:
            state = 'done' if p['ok'] else ('failed' if r[2]>=self.max_attempts else 'pending')
            result = json.dumps({**p,'accepted_at':now})
            self.db.execute('UPDATE tasks SET status=?,result=? WHERE id=?',(state,result,p['task_id']))
            self.change('running',state)
        return {'accepted':True}

    def writer(self):
        try:
            self.db = sqlite3.connect(self.path, timeout=60)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=NORMAL')
            self.db.execute('PRAGMA wal_autocheckpoint=0')
            self.db.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY,task TEXT,status TEXT,token TEXT,deadline REAL,attempts INTEGER,result TEXT)')
            self.db.execute('CREATE INDEX IF NOT EXISTS tasks_pending_id ON tasks(status,id)')
            self.db.execute('CREATE INDEX IF NOT EXISTS tasks_status_deadline ON tasks(status,deadline)')
            self.counts = dict(self.db.execute('SELECT status,count(*) FROM tasks GROUP BY status'))
            self.db.commit()
            self.published = self.counts.copy()
        except Exception as e:
            self.error = e; self.ready.set(); return
        self.ready.set()
        next_expire = 0
        while True:
            item = self.inbox.get()
            if item is None: break
            batch = [item]
            # Bound coalescing delay; commit up to 128 updates together.
            end = time.monotonic()+.002
            while len(batch)<128:
                try: batch.append(self.inbox.get(timeout=max(0,end-time.monotonic())))
                except queue.Empty: break
            old_counts = self.counts.copy()
            replies = []
            try:
                with self.db:
                    self.db.execute('BEGIN IMMEDIATE')
                    now = time.time()
                    if now>=next_expire: self.expire(now); next_expire=now+1
                    for op,p,f in batch:
                        self.db.execute('SAVEPOINT request')
                        before = self.counts.copy()
                        try:
                            response = self.apply(op,p,now)
                            self.db.execute('RELEASE request')
                            replies.append((f,response,None))
                        except Exception as e:
                            self.db.execute('ROLLBACK TO request');self.db.execute('RELEASE request')
                            self.counts = before;replies.append((f,None,e))
                with self.metrics_lock: self.published = {k:v for k,v in self.counts.items() if v}
                for f,r,e in replies:
                    if e: f.set_exception(e)
                    else: f.set_result(r)
            except Exception as e:
                self.counts = old_counts
                for _,_,f in batch:
                    if not f.done(): f.set_exception(e)
        self.db.close()

    def close(self):
        # Caller first drains HTTP requests, then stops writer.
        self.inbox.put(None);self.thread.join()

class Server(ThreadingHTTPServer):
    request_queue_size = 512
    daemon_threads = False

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db',required=True);p.add_argument('--restore',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--host',default='0.0.0.0')
    p.add_argument('--port',type=int,default=18818);a=p.parse_args()
    db=Path(a.db);db.parent.mkdir(parents=True,exist_ok=True)
    if not db.exists():
        tmp=db.with_suffix('.restore');source=Path(a.checkpoint) if Path(a.checkpoint).exists() else Path(a.restore);shutil.copyfile(source,tmp);os.replace(tmp,db)
    q=Queue(db);stop=threading.Event()
    def checkpoint():
        local=db.with_suffix('.snapshot');out=sqlite3.connect(local)
        src=sqlite3.connect(db,timeout=60)
        try:
            # Pin a WAL read snapshot: concurrent writers must not restart backup steps.
            src.execute('BEGIN');src.execute('SELECT count(*) FROM sqlite_master').fetchone()
            src.backup(out,pages=4096,sleep=.01)
        finally: out.close();src.close()
        dest=Path(a.checkpoint);dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix('.tmp');shutil.copyfile(local,tmp);os.replace(tmp,dest);local.unlink()
        dest.with_suffix('.status.json').write_text(json.dumps({'unix':time.time(),**q.status()}))
    def backups():
        while not stop.wait(300):
            try: checkpoint()
            except Exception as e: print('CHECKPOINT_ERROR',repr(e),flush=True)
    def wal_flush():
        conn=sqlite3.connect(db,timeout=30)
        while not stop.wait(5):
            try:conn.execute('PRAGMA wal_checkpoint(PASSIVE)')
            except Exception as e:print('WAL_CHECKPOINT_ERROR',repr(e),flush=True)
        conn.close()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=8*1024*1024: raise ValueError('Invalid request size')
                result=q.call(self.path.strip('/'),json.loads(self.rfile.read(size)));code=200
            except Exception as e: result={'error':str(e)};code=400
            body=json.dumps(result).encode();self.send_response(code)
            self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def log_message(self,*args):pass
    server=Server((a.host,a.port),Handler)
    def quit(*_):stop.set();threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,quit);signal.signal(signal.SIGINT,quit)
    threads=[threading.Thread(target=backups,daemon=True),threading.Thread(target=wal_flush,daemon=True)]
    for t in threads:t.start()
    print('SERVER_READY',a.host,a.port,q.status(),flush=True)
    try:server.serve_forever()
    finally:
        stop.set();server.server_close()
        for t in threads:t.join()
        checkpoint();q.close()
if __name__=='__main__':main()
