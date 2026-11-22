"""Ejecución reproducible sin interfaz. Ejemplo:
python -m generabuild.cli --site guanajuato8 --floors 9 --preview-only
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .storage import site_path,load_metadata
from .scenario import config_from_metadata,prepare_run,preview_masks
from .generation import cuda_available,load_pipeline,generate_one
from .enhancement import apply_enhancements
from .evaluation import evaluate_run


def main(argv=None):
    parser=argparse.ArgumentParser(description='GeneraBuild FR2 CLI — reproducible urban scenario')
    parser.add_argument('--site',required=True,help='Nombre de sitio bajo workspace/sites/ o ruta existente')
    parser.add_argument('--floors',type=int,default=9,choices=[8,9,10])
    parser.add_argument('--mode',default='ampliacion',choices=['ampliacion','completo'])
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--views',nargs='+',type=int,default=[1,2,3,4])
    parser.add_argument('--preview-only',action='store_true',help='Crear máscaras y previsualizaciones sin GPU')
    parser.add_argument('--no-enhancement',action='store_true')
    parser.add_argument('--no-roof',action='store_true')
    parser.add_argument('--context-occlusion',action='store_true',help='EXPERIMENTAL: aplicar prismas vecinos como oclusores; por defecto desactivado')
    parser.add_argument('--config',type=Path,help='Config JSON o YAML, sobreescribe opciones anteriores')
    args=parser.parse_args(argv)
    site=Path(args.site)
    if not site.is_dir():site=site_path(args.site)
    if not (site/'calibration_multiview.json').is_file():
        parser.error(f'No se encuentra calibration_multiview.json en {site}. Importa primero el ZIP de Colab.')
    meta=load_metadata(site)
    config=config_from_metadata(meta,floors=args.floors,mode=args.mode,seed=args.seed,views=args.views,
                                photo_finish=not args.no_enhancement,roof_finish=not args.no_roof)
    if args.config:
        import yaml
        custom=yaml.safe_load(args.config.read_text(encoding='utf-8'))
        if not isinstance(custom,dict): raise ValueError('Config debe ser un objeto YAML/JSON')
        # merge shallow con sección postprocessing
        extra=dict(custom)
        config['postprocessing'].update(extra.pop('postprocessing',{}))
        config.update(extra)
    run_dir=prepare_run(site,config)
    if args.context_occlusion:
        from .context_occlusion import apply_context_occlusions_to_run
        report=apply_context_occlusions_to_run(run_dir,site)
        print('Oclusión 2.5D:',json.dumps(report,ensure_ascii=False))
    print('Scenario:',run_dir)
    for r in preview_masks(run_dir,config['mode']):
        print(f"  Vista {r['view']}: {r['work_editable']:,} píxeles editables")
    if args.preview_only:
        print('Vista previa creada (no se ejecutó difusión).')
        return 0
    if not cuda_available():
        raise RuntimeError('Se requiere PyTorch con CUDA; prueba --preview-only para validar el escenario en CPU.')
    pipe=load_pipeline()
    base={i:generate_one(run_dir,i,pipe,config) for i in config['views']}
    final,reports=apply_enhancements(run_dir,base,config)
    (run_dir/'enhancement_reports.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    manifest={'base_images':{str(k):str(Path(v).relative_to(run_dir)) for k,v in base.items()},
              'final_images':{str(k):str(Path(v).relative_to(run_dir)) for k,v in final.items()}}
    (run_dir/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    result=evaluate_run(run_dir,final,config)
    print('Resultados:',json.dumps({'run':str(run_dir),'context_pass':result['context_preservation_pass']},ensure_ascii=False))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
