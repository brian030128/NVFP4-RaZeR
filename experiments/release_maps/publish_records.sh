#!/bin/bash
# Copy the release records (not the maps) into the NVFP4-RaZeR worktree of branch release-maps:
#   results/release_maps/{README.md, REPORT.md, summary.json}
#   results/release_maps/<model>/{MODEL_CARD.md, NOTICE?, records/<unit>/*.json, ppl/*.json, prep/*.json}
#   experiments/release_maps/: the scripts
set -eu
B=/home/dev/n16k64_campaign/fqrel; REL=/home/dev/flipquant_release; R=/home/dev/n16k64_campaign/relmaps/wt
D=$R/results/release_maps; E=$R/experiments/release_maps
mkdir -p $D $E
cp $B/release.py $B/report.py $B/chain2.sh $B/chain3.sh $B/watch_tree.py $B/remeasure.py $B/measure.py \
   $B/measure_vmhwm.py $B/prep_run.py $B/compare_prep.py $B/publish_records.sh $E/
cp $B/RECORDS_README.md $D/README.md
cp $REL/REPORT.md $REL/summary.json $D/
for m in $(ls $REL); do
  [ -d $REL/$m/records ] || continue
  mkdir -p $D/$m/ppl
  cp $REL/$m/README.md $D/$m/MODEL_CARD.md
  [ -f $REL/$m/NOTICE ] && cp $REL/$m/NOTICE $D/$m/NOTICE
  for u in $(ls $REL/$m/records); do
    mkdir -p $D/$m/records/$u
    for f in run.json reproduction.json trainer_report.json remeasure.json; do
      [ -f $REL/$m/records/$u/$f ] && cp $REL/$m/records/$u/$f $D/$m/records/$u/
    done
    # the live process-tree measurement (watch_tree.py) of the release run, and of its measured re-run
    for f in ${m}_${u}.json ${m}_${u}_remeasure.json; do
      [ -f $B/measure/$f ] && cp $B/measure/$f $D/$m/records/$u/watch_${f#${m}_${u}}
    done
  done
  cp $REL/$m/ppl/*.json $D/$m/ppl/
  if [ -f $B/prep_measure/$m.measure.json ]; then
    mkdir -p $D/$m/prep
    cp $B/prep_measure/$m.measure.json $D/$m/prep/measure.json
    cp $B/prep_measure/$m.compare.json $D/$m/prep/compare.json
  fi
done
du -sh $D $E
