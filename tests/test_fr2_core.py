import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from scripts import create_demo_site
import generabuild.storage as storage
from generabuild.scenario import config_from_metadata,prepare_run,preview_masks
from generabuild.enhancement import apply_enhancements
from generabuild.evaluation import evaluate_run
from generabuild.export import zip_folder
from generabuild.geometry import validate_project
from gb21_extension_mask import extension_work_mask


@pytest.fixture
def demo(tmp_path,monkeypatch):
    # La función importó site_path en su propio módulo.
    monkeypatch.setattr(create_demo_site,'site_path',lambda name:tmp_path/name)
    project=create_demo_site.create_demo('demo',size=(680,480))
    return project


def test_project_roundtrip_and_multipart_cameras(demo):
    meta=storage.load_metadata(demo)
    assert len(meta['views'])==4
    assert meta['views'][3]['fov_horizontal_deg']==110
    rows=validate_project(demo,meta)
    assert all(row['mask_reproducible_exact'] for row in rows)
    assert all(row['rmse_px'] < 2 for row in rows)


def test_isolated_parametric_runs_preserve_calibration(demo):
    meta=storage.load_metadata(demo)
    original=(demo/'calibration_multiview.json').read_bytes()
    outputs=[]
    for floors in (8,9,10):
        conf=config_from_metadata(meta,floors=floors,mode='ampliacion',views=[1,2])
        run=prepare_run(demo,conf,run_id=f'height_{floors}')
        assert storage.load_metadata(run)['num_floors']==floors
        masks=preview_masks(run,'ampliacion')
        assert all(m['work_editable']>0 for m in masks)
        assert all(m['work_editable']<=m['total_editable'] for m in masks)
        outputs.append([m['work_editable'] for m in masks])
    assert outputs[0] != outputs[-1]
    assert (demo/'calibration_multiview.json').read_bytes()==original


def test_legacy_optional_enhancement_and_context(demo):
    meta=storage.load_metadata(demo)
    conf=config_from_metadata(meta,floors=9,mode='ampliacion',views=[1,3],
                              front_windows=3,side_windows=4,photo_finish=True,roof_finish=True,
                              roof_equipment=0)
    run=prepare_run(demo,conf,run_id='finish_integration')
    runmeta=storage.load_metadata(run)
    base={}
    for i in (1,3):
        view=runmeta['views'][i-1]
        image=np.array(Image.open(run/view['image_relative']).convert('RGB'))
        mask=np.array(Image.open(run/view['mask_relative']).convert('L'))
        work=extension_work_mask(mask,runmeta,i,True)>127
        image[work]=[163,162,157]
        path=run/'generated_realistic'/f'Vista_{i}_9pisos_ampliacion_s1234.png'
        path.parent.mkdir(exist_ok=True,parents=True)
        Image.fromarray(image).save(path)
        base[i]=path
    final,reports=apply_enhancements(run,base,conf)
    assert len(final)==2
    assert 'facade_1' in reports and 'roof' in reports
    report=evaluate_run(run,final,conf)
    assert report['context_preservation_pass']
    assert all(r['context_error_pct']==0 for r in report['views'])
    assert all(Path(p).is_file() for p in final.values())
    assert (run/'evaluation.json').is_file()


def test_zip_import_from_colab_structure(demo,tmp_path,monkeypatch):
    raw=zip_folder(demo)
    monkeypatch.setattr(storage,'WORKSPACE',tmp_path/'imported')
    target=storage.import_colab_zip(raw,'test_import')
    assert len(storage.load_metadata(target)['views'])==4
    assert all(p.exists() for p in storage.image_paths(target,storage.load_metadata(target)))
    assert all(r['mask_reproducible_exact'] for r in validate_project(target,storage.load_metadata(target)))
    with pytest.raises(FileExistsError):
        storage.import_colab_zip(raw,'test_import')
