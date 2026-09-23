"""Copy only existing full8 Arrow cache payloads; never mutate old caches.

Byte checks and Arrow counts establish a stable copied cache, not equality of
new evaluation prompts. Baseline sample/prompt identity remains a GPU-run gate.
No model import, dataset download, selector change or inference occurs here.
"""
import shutil
import pyarrow as pa
from common import *

NAMES=['Rowan___hellaswag','allenai___ai2_arc','allenai___openbookqa',
       'allenai___winogrande','aps___super_glue','baber___piqa','cais___mmlu']
RUNTIME=REPO/'research_runs/mixfp4_selector_characterization_20260921T133858Z'

def main():
    start=now();source=PRIMARY/'cache/hf/datasets';dest=RUNTIME/'cache/accuracy_datasets'
    dest.mkdir(parents=True,exist_ok=True);files=[];tables=[]
    for name in NAMES:
        base=source/name;assert base.is_dir(),base
        for original in sorted(base.rglob('*')):
            if not original.is_file() or not (original.suffix=='.arrow' or original.name=='dataset_info.json'):continue
            rel=original.relative_to(source);copy=dest/rel;before=sha(original)
            if copy.exists():assert sha(copy)==before,('Existing accuracy snapshot differs; preserve and diagnose',copy)
            else:
                copy.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(original,copy)
            after=sha(original);assert before==after==sha(copy),original
            files.append(dict(source=str(original.relative_to(REPO)),path=str(copy.relative_to(REPO)),
                sha256=before,bytes=copy.stat().st_size,source_before_after_equal=True))
        for info in sorted((dest/name).rglob('dataset_info.json')):
            meta=load(info)
            for split,details in meta['splits'].items():
                path=info.parent/f'{meta["dataset_name"]}-{split}.arrow'
                assert path.exists(),path
                with pa.memory_map(str(path),'r') as f:
                    reader=pa.ipc.open_stream(f);rows=sum(batch.num_rows for batch in reader)
                assert rows==details['num_examples'],(path,rows,details['num_examples'])
                tables.append(dict(dataset=name,config=meta.get('config_name'),split=split,rows=rows,
                    path=str(path.relative_to(REPO)),metadata_sha256=sha(info)))
    report=dict(status='copied_and_byte_verified',checked_utc=start,destination=str(dest.relative_to(REPO)),
        repairs=[dict(attempt=1,error='Assumed builder_name was Arrow filename prefix; parquet builder stores hellaswag-*.arrow',
            resolution='Use dataset_info.dataset_name, confirmed against original filenames. No rows or hashes changed.')],
        files=files,tables=tables,total_bytes=sum(x['bytes'] for x in files),
        qualification='Existing cache bytes/schema/counts only. Original inputs unchanged; locks excluded. Prompt/template/sample identity must still match frozen accuracy records before reuse.')
    jsonout(OUT/'results/ACCURACY_CACHE_AUDIT.json',report)
    log('accuracy-cache','python scripts/prepare_accuracy_cache.py',start,[OUT/'results/ACCURACY_CACHE_AUDIT.json'])
    print('Verified cache files',len(files),'tables',len(tables),'bytes',report['total_bytes'])

if __name__=='__main__':main()
