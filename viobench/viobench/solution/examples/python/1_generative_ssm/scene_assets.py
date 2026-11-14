"""Remove empty GLB primitives before Assimp can abort the collector process."""
import json
from pathlib import Path
import shutil
import struct


def prepare_scene(reference,directory):
    if isinstance(reference,dict) or (isinstance(reference,str) and reference.startswith('conta:')):
        import conta
        reference=conta.resolve(reference)
    path=Path(reference)
    with path.open('rb') as source:
        header=source.read(12)
        if header[:4]!=b'glTF':return str(path),0
        if len(header)!=12:raise ValueError('Truncated GLB header')
        magic,version,total=struct.unpack('<4sII',header)
        if version!=2 or total!=path.stat().st_size:raise ValueError('Invalid GLB version/length')
        chunk=source.read(8)
        if len(chunk)!=8:raise ValueError('Missing GLB JSON chunk')
        length,kind=struct.unpack('<I4s',chunk)
        if kind!=b'JSON':raise ValueError('GLB must begin with JSON')
        document=json.loads(source.read(length))
        accessors=document.get('accessors',[])
        meshes=[];indices={};removed=0
        for index,mesh in enumerate(document.get('meshes',[])):
            primitives=[]
            for primitive in mesh.get('primitives',[]):
                accessor=primitive.get('indices',primitive.get('attributes',{}).get('POSITION'))
                if accessor is not None and accessors[accessor]['count']==0:
                    removed+=1
                else:primitives.append(primitive)
            if primitives:
                indices[index]=len(meshes)
                meshes.append(dict(mesh,primitives=primitives))
            elif not mesh.get('primitives'):removed+=1
        if not removed:return str(path),0
        if not meshes:raise ValueError('GLB contains no nonempty meshes')
        document['meshes']=meshes
        for node in document.get('nodes',[]):
            if 'mesh' not in node:continue
            if node['mesh'] in indices:node['mesh']=indices[node['mesh']]
            else:
                node.pop('mesh');node.pop('weights',None)
        encoded=json.dumps(document,separators=(',',':'),ensure_ascii=False).encode('utf-8')
        encoded+=b' '*((-len(encoded))%4)
        destination=Path(directory)/'scene.glb'
        # All buffers, textures, transforms and nonempty primitives stay intact.
        # Buffer-view offsets are relative to BIN, whose bytes do not change.
        with destination.open('wb') as output:
            output.write(struct.pack('<4sII',magic,version,total-length+len(encoded)))
            output.write(struct.pack('<I4s',len(encoded),b'JSON'));output.write(encoded)
            shutil.copyfileobj(source,output,length=8*2**20)
    return str(destination),removed
