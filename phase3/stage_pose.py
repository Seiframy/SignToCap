"""Stage the large cached RTMW file as restartable single-PUT objects."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

SIZE = 8 * 1024**2

def main(cli, source, key, volume):
    base = [cli, "--profile","signgpt-volume","--region","EUR-IS-1","--endpoint-url",
            "https://s3api-eur-is-1.runpod.io/","--cli-read-timeout","90","--no-cli-pager","--output","json"]
    listing=json.loads(subprocess.check_output(base+["s3api","list-objects-v2","--bucket",volume,"--prefix",key+".parts/"]))
    remote={o["Key"]:o for o in listing.get("Contents") or []}
    def put(item):
        path, name = item
        existing=remote.get(name)
        checksum=hashlib.md5(path.read_bytes(),usedforsecurity=False).hexdigest()
        if existing and existing["ETag"].strip('"')==checksum and existing["Size"]==path.stat().st_size:
            return
        subprocess.run(base+["s3api","put-object","--bucket",volume,"--key",name,"--body",str(path)],
                       check=True,stdout=subprocess.DEVNULL)
        print("Staged "+name.rsplit("/",1)[-1],flush=True)
    with TemporaryDirectory() as directory, open(source,"rb") as stream:
        pieces=[]
        while data:=stream.read(SIZE):
            path=Path(directory)/f"{len(pieces):03d}"
            path.write_bytes(data)
            pieces.append((path,key+".parts/"+path.name))
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(put,pieces))

if __name__=="__main__":
    main(*sys.argv[1:])
