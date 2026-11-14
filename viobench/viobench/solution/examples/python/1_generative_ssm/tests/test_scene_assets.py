"""Repair empty GLB primitives without changing rendered/collision geometry."""
import json
from pathlib import Path
import struct
import sys

import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from scene_assets import prepare_scene


def make_glb(path,document,binary=b'abcdefgh'):
    encoded=json.dumps(document).encode();encoded+=b' '*((-len(encoded))%4)
    path.write_bytes(struct.pack('<4sII',b'glTF',2,12+8+len(encoded)+8+len(binary))+
        struct.pack('<I4s',len(encoded),b'JSON')+encoded+struct.pack('<I4s',len(binary),b'BIN\0')+binary)


def test_empty_index_mesh_removed_and_node_indices_remapped_without_changing_bin(tmp_path):
    document=dict(asset=dict(version='2.0'),accessors=[dict(count=3),dict(count=0),dict(count=9)],
        meshes=[dict(primitives=[dict(indices=0)]),dict(primitives=[dict(indices=1)]),
                dict(primitives=[dict(indices=2),dict(indices=1)])],
        nodes=[dict(mesh=0),dict(mesh=1,children=[2],translation=[1,2,3]),dict(mesh=2)],
        scenes=[dict(nodes=[0,1])],scene=0)
    source=tmp_path/'original.glb';make_glb(source,document);before=source.read_bytes()
    directory=tmp_path/'scratch';directory.mkdir()
    result,removed=prepare_scene(source,directory)
    assert removed==2 and source.read_bytes()==before
    data=Path(result).read_bytes();magic,version,total=struct.unpack('<4sII',data[:12])
    length,kind=struct.unpack('<I4s',data[12:20]);clean=json.loads(data[20:20+length])
    assert magic==b'glTF' and version==2 and total==len(data) and kind==b'JSON'
    assert len(clean['meshes'])==2 and clean['meshes'][1]['primitives']==[dict(indices=2)]
    assert clean['nodes']==[dict(mesh=0),dict(children=[2],translation=[1,2,3]),dict(mesh=1)]
    assert data[20+length:]==struct.pack('<I4s',8,b'BIN\0')+b'abcdefgh'
    assert clean['accessors']==document['accessors']


def test_valid_scene_uses_original_path_and_needs_no_copy(tmp_path):
    source=tmp_path/'valid.glb'
    make_glb(source,dict(meshes=[dict(primitives=[dict(indices=0)])],accessors=[dict(count=3)]))
    result,removed=prepare_scene(source,tmp_path/'unused')
    assert result==str(source) and removed==0 and not (tmp_path/'unused').exists()


def test_completely_empty_scene_fails_before_native_loader(tmp_path):
    source=tmp_path/'empty.glb'
    make_glb(source,dict(meshes=[dict(primitives=[])],accessors=[]))
    with pytest.raises(ValueError,match='no nonempty meshes'):prepare_scene(source,tmp_path)
