import json
from pathlib import Path
import pytest
from soarm_training import pipeline
def test_export_is_immutable_and_hashes_processors(tmp_path):
    source=tmp_path/'checkpoint';source.mkdir()
    for name in ('config.json','model.safetensors','policy_preprocessor.json','policy_postprocessor.json'):(source/name).write_text('{}')
    profile=Path(__file__).resolve().parents[1]/'config/export.example.yaml'
    manifest=pipeline.export(source,profile,tmp_path/'export')
    assert len(manifest['files'])==4
    assert manifest['evaluation']['task_outcome']=='NOT_PHYSICALLY_EVALUATED'
    with pytest.raises(FileExistsError):pipeline.export(source,profile,tmp_path/'export')
def test_incomplete_checkpoint_rejected(tmp_path):
    with pytest.raises(ValueError,match='Incomplete'):pipeline.export(tmp_path,Path(__file__).resolve().parents[1]/'config/export.example.yaml',tmp_path/'out')
