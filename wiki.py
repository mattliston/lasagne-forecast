import subprocess
import json

p = subprocess.Popen(['curl','-XGET','https://www.quandl.com/api/v3/datatables/WIKI/PRICES?qopts.export=true&api_key=zzfSrwzFo8jcJa9p5rxx','-H','Content-Type: application/json'],stdout=subprocess.PIPE)
(stdoutdata, stderrdata) = p.communicate()
d = json.loads(stdoutdata)
print json.dumps(d, sort_keys=True, indent=4, separators=(',', ': ')) # pretty print
p = subprocess.Popen(['wget',d['datatable_bulk_download']['file']['link'],'-O','wiki.zip'])
(stdoutdata, stderrdata) = p.communicate()
p = subprocess.Popen(['unzip','wiki.zip'])
(stdoutdata, stderrdata) = p.communicate()
