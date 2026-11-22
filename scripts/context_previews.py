"""Proyecta un entorno 2.5D opcional sin depender de PySide6."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, json
import cv2
import numpy as np
from PIL import Image
from generabuild.context_2p5d import load_context,world_to_image
from generabuild.context_occlusion import neighbor_occlusion_mask
from generabuild.storage import load_metadata,image_paths
from generabuild.geometry import box_vertices


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--floors',type=int,default=9,choices=[8,9,10])
    parser.add_argument('--output',type=Path,default=None)
    a=parser.parse_args(argv)
    meta=load_metadata(a.site);buildings=load_context(a.site)
    if not buildings:parser.error('No existe ningún vecino 2.5D. Anótalo en GeneraBuild Studio.')
    ref=meta['reference_box_m']
    target=box_vertices(ref['W'],ref['D'],a.floors*meta['floor_height_m'])
    output=a.output or a.site/'diagnostics'/'context_2p5d'
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,(p,v) in enumerate(zip(image_paths(a.site,meta),meta['views']),1):
        rgb=np.asarray(Image.open(p).convert('RGB')).copy()
        BGR=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        P=np.asarray(v['P'],float)
        for b in buildings:
            if not b.enabled:continue
            try:
                uv=np.rint(world_to_image(P,b.vertices())).astype(np.int32)
            except ValueError:continue
            n=len(b.footprint)
            cv2.polylines(BGR,[uv[:n],uv[n:]],True,(124,224,59),2)
            for j in range(n):cv2.line(BGR,tuple(uv[j]),tuple(uv[j+n]),(124,224,59),2)
        cover=neighbor_occlusion_mask(P,target,buildings,Image.open(p).size)
        cv2.imwrite(str(output/f'vista_{i}_contexto.png'),BGR)
        cv2.imwrite(str(output/f'vista_{i}_occlusion.png'),cover)
        rows.append({'view':i,'occlusion_pixels':int(np.count_nonzero(cover))})
    (output/'report.json').write_text(json.dumps({'views':rows,'experimental':True},indent=2),encoding='utf-8')
    print('Diagnóstico 2.5D creado en:',output)


if __name__=='__main__': main()
