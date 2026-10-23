import json,urllib.request
r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30296/v1/completions",data=json.dumps({"model":"dsv41-flash","prompt":"Once upon a time","max_tokens":60,"temperature":0}).encode(),headers={"Content-Type":"application/json"}))
print(r.read()[:150])