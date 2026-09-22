"""Render the final report as Feishu XML with native tables and local figures."""
import re,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'camera pose消融试验记录报告.md';dest=ROOT/'draft_895dfe1a_folder/draft.xml'
def inline(s):
 pattern=r'(\*\*.*?\*\*|`[^`]+`|\[[^\]]+\]\([^\)]+\))';out=[];last=0
 for m in re.finditer(pattern,s):
  out.append(html.escape(s[last:m.start()]));t=m.group()
  if t.startswith('**'):out.append('<b>'+html.escape(t[2:-2])+'</b>')
  elif t.startswith('`'):out.append(html.escape(t[1:-1]))
  else:
   link=re.match(r'\[([^\]]+)\]\(([^\)]+)\)',t);label,url=link.groups()
   if url.startswith('https://'):out.append('<a href="'+html.escape(url,quote=True)+'">'+html.escape(label)+'</a>')
   else:out.append(html.escape(label)+'（'+html.escape(url)+'）')
  last=m.end()
 out.append(html.escape(s[last:]));return ''.join(out)

def main():
 lines=source.read_text().splitlines();out=[];i=0
 while i<len(lines):
  line=lines[i].strip();i+=1
  if not line:continue
  if line.startswith('# '):out.append('<title>'+html.escape(line[2:])+'</title>');continue
  if line.startswith('##'):
   m=re.match(r'(#+) (.*)',line);level=len(m[1])-1;label=re.sub(r'^(?:[一二三四五六七八九十]+、|\d+(?:\.\d+)*\s*)','',m[2]);out.append(f'<h{level} seq="auto">'+inline(label)+f'</h{level}>');continue
  if line.startswith('!['):
   m=re.match(r'!\[([^\]]*)\]\(([^)]+)\)',line);assert (ROOT/m[2]).is_file();out.append('<img path="@./'+html.escape(m[2],quote=True)+'" caption="'+html.escape(m[1],quote=True)+'"/>');continue
  if line.startswith('|'):
   rows=[line]
   while i<len(lines) and lines[i].strip().startswith('|'):rows.append(lines[i].strip());i+=1
   cells=[[v.strip() for v in row.strip('|').split('|')] for row in rows];out.append('<table><thead><tr>')
   for c in cells[0]:out.append('<th background-color="light-gray"><p>'+inline(c)+'</p></th>')
   out.append('</tr></thead><tbody>')
   for row in cells[2:]:
    out.append('<tr>')
    for c in row:out.append('<td><p>'+inline(c)+'</p></td>')
    out.append('</tr>')
   out.append('</tbody></table>');continue
  if line.startswith('- '):
   items=[line[2:]]
   while i<len(lines) and lines[i].startswith('- '):items.append(lines[i][2:]);i+=1
   out.append('<ul>'+''.join('<li>'+inline(x)+'</li>' for x in items)+'</ul>');continue
  if re.match(r'^\d+\. ',line):
   items=[re.sub(r'^\d+\. ','',line)]
   while i<len(lines) and re.match(r'^\d+\. ',lines[i]):items.append(re.sub(r'^\d+\. ','',lines[i]));i+=1
   out.append('<ol>'+''.join('<li>'+inline(x)+'</li>' for x in items)+'</ol>');continue
  out.append('<p>'+inline(line)+'</p>')
 out+=['<h2 seq="auto">数据附件与复现位置</h2>','<p>本报告中的相对文件路径均相对以下工作目录；关键CSV和审计结果同时作为附件提供，便于不登录计算节点也能核对数据。</p>','<p>'+html.escape(str(ROOT))+'</p>']
 for path,name in [('results/omega_window_study/summary.csv','fps_overlap消融汇总.csv'),('results/lerobot_8gpu_4fps_ov36/per_gpu.csv','早期八卡每卡吞吐.csv'),('results/lerobot_pipeline_ab/paired.csv','流水线2048条配对结果.csv'),('results/lerobot_pipeline_ab/summary.json','流水线对照汇总与数值审计.json')]:
  assert (ROOT/path).is_file();out.append('<source path="@./'+path+'" name="'+name+'"/>')
 text='\n'.join(out);assert 'batch' not in text.lower();dest.write_text(text);print(dest)
if __name__=='__main__':main()
