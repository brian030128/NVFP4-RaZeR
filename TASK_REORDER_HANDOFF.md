# Native full-model measurement update — 2026-09-20 09:35 UTC

No active jobs. Latest question: why final MLP only? Practical cheap cached-suffix
search and limited permutation scope, not proven optimality. Wider Qwen scopes
had mixed PPL / failed subsequent gates; no exhaustive Llama scope search.
Other layers retain raw256 MixFP4. Rationale added to MIXFP4_REPORT/source.

Full native Llama diagnostic timing completed in406751 and406828, oneGB200 at a
time, gov113008. Monitors handled; receipt406828 recorded. Same-session queue
failed earlier; attached monitor output is the completion mechanism used here.
406751 cold decode pilot retained.406828 uses two full-request warmups perpolicy,
all6policy orders, GCdisabled, and reuses672 passed operator+activation audits
with hash/AST guards; repeats224 weight audits. Prompt128+32decode arranged
request overhead+0.531%±0.526%(2SE), median915.747ms vsbase911.380ms. Prompt2048
varies substantially across ALL policies (decode27–53ms/token): INCONCLUSIVE.
Do not cherry-pick samples or claim reliable long-context overhead. Actual
heterogeneous finalMLP GEMM-only timings are also saved, with run variability.

IMPORTANT: This is explicitly a prospective DIAGNOSTIC latency arm. Original
full-model output gates FAILED for mixed policies, remain failed, and native
PPL is UNMEASURED. No quality candidate/gate changed. Corrected activation
producer passes all actual codes/scales; native repeat forwards bitwise.
Native vsFP64 dot all672 operators relative<.005. Baseline matches epilogue-order
FP32 fullreference bitwise, mixed policies diverge ~10.8% final logits: tinyGEMM
rounding differences amplified through laterFP4quantization. Single-layer2 replay
406741 proves identical inputs, only0–4 projectionBF16diffs, layerrel4.586e-6.
Don't imply old fakequantPPL is demonstrated by native backend.

Evidence: results/task_reorder/full_model_20260920/{plan.json,IMPLEMENTATION.md,
llama_406751, llama_406828, kernel_406633, layerdiff_406741}; see directory
for exact names. Library ~/.cache/mixfp4-model-runtime/build/libmixfp4_model.so,
ARMvenv under same runtime root. Original calibration/layouts unchanged. Sibling
../mixfp4 unchanged. No Qwen fullmodel integration (HF5/N48 support needed).
No new job justified merely by notification replay. Further timing work needs
addressing measured host/eager variability; don't rerun blindly.

Historical entries below are superseded by this update.

# Active native full-model debugging — 2026-09-20 08:45 UTC

Latest user asks why final MLP only; answered in commentary, added rationale to
report (not pushed). Continue earlier full-model latency request, no new quality
search. Native full-model timing STILL UNMEASURED: gates stopped every attempt.

ACTIVE406633 kernelcompile+extendedcheck, oneGB200gov113008, attachedmonitor
63661, outputs results/task_reorder/full_model_20260920/kernel_406633. Native
library /home/u4320956/.cache/mixfp4-model-runtime/build/libmixfp4_model.so.
ARM env ready torch2.9.0+cu130/transformers4.57.3 at runtime/venv. Workquota issue
resolved by moving ONLY new env/cache to home, hardlinkinstall. Maxactual1GPU.
No other jobs active. Keep attached monitor; prior codex queue delivery failed.

CRITICAL root cause now identified using reusable2-layer case406608 (12sec):
/work/u4320956/b200/activation_case_406608/layers_1_self_attn_{q,v}_proj.pt.
Input32x4096 and packedweights/scales captured; don't reload model for debugging.
Native activation has133 differingcodebytes/37scales/478decodedvalues vsPython;
GEMM agrees with actual native dumped operands (rel.00166), soGEMM iscorrect.
PyTorch scalar division multiplies rounded reciprocal (official v2.9.0
aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu). Native division gaveglobalbit
0x3b018618 instead ofreference0x3b018619, crossingFP8thresholds. FIX in406633:
global=max*(1.f/2688.f), scale6=mx*(1.f/6.f), sum errors ascendingXOR1,2,4,8
matchingTorchadjacenttree; broadcast one lane0 election to16codes/sharedscale.
Both native/quantize_permutation.cu and duplicateaudit fixed; modelalpha fixed.
Small checks include3millionBF16values AND savedrealcase bitwise codes/scales +
GEMM numeric. Do NOTrun another fullmodel until406633passes.

Prior406525 passed smallrandomkernel8checks.406530 failed tokenizerremotequery
(workarounduseexactcachedpath).406546 packed224weightsbitwisepassed but fullmodel
native-vsBF16fake failed12.412%.406553 diagnostic448operatorchecks43failed,
max2.145%; no timings.406575 broadcast-only quantfix passed3million synthetic
values but DIDNOTfix realinput.406591 same12.412%BF16/12.666%exact-reference
fullmodel failure, no timings.406587 snapshotmissingimportance failedbeforeload,
fixedclosure inbatch. Receipts recorded all terminals. Don't relabel failedgates.

scripts/benchmark_native_llama.py now requires --kernel-gate directory with
passed report.json and library.json SHA256. After406633passes write library.json
using currentlibraryhash, then submit NATIVE_KERNEL_GATE=absolute/kernel_406633
sbatch slurm/gb200_full_llama.sbatch. It snapshots importclosure perjob.
Frozen engineering gate now uses SAME native arithmetic: every224 operator<.005
relative toexactdecodedFP32, fullmodellogits<.02 vsFP32decode/BF16outputreference;
BF16fakegap reportedseparately (oldfailurepreserved), NOquality/PPLpromotion.
Oldfullmodelgate protocol+prospectiverevision in plan.json. Fullrun thenmeasures
actualheterogeneousGEMMonly (3finalMLPprojsxM1/128/2048) andwholemodelprefill128/2048
+32cacheddecode,3policiesbase/raw/arranged,5pairedreps. Columnsproducerfused;
rowsrestoredseparatekernel. Iffailsagain, save reusableinput cases beforemorejobs.

scripts/summarize_native_llama.py prepared for completed report; not run.
Need update MIXFP4_REPORT+source, status,handoff,costledger andpush when finished.
All new scripts/nativefiles uncommitted; many unrelatedpreexisting dirtyfiles,
stageexplicitpaths. Reportedit uses python3 build_mixfp4_report.py --update-arranging.
Qwenfullmodel notintegratedyet (HF5.16.1 +N48linears; canpad48to256forfixedtile,
sharedbaselinecostlabel); don'tclaimQwenlatency. Usercurrentquestionanswer: scope
practical/cheapcachedsuffix, smalloverhead, broaderQwentradeoffs/failedgates;
noexhaustiveLlamalayersearch, notprovenoptimal; otherlayersstillrawMixFP4.

# Active native full-model benchmark — 2026-09-20 08:14 UTC

User explicitly requested full-model latency measurement; latest question asks why
only final MLP rearranged. Explained scope is practical, not proven optimal;
matched wider Qwen scope traded Wiki/C4 and later extensions failed gates.
Added scope rationale to report source and MIXFP4_REPORT (not yet pushed).

GB200 ARM setup406442 failed work quota (78s);406471 moved downloaded cache to
/home successfully but failed uv refusing partial venv (55s); dependent406483
cancelled with zero allocation. Receipts recorded. Repaired setup406496 PENDING,
attached monitor session96877; compile+kernel audit406498 depends afterok,
attached monitor48665. ALWAYS handle monitors; queue delivery failed previously.
At most4 GPUs, gov113008; currently at most1 for this sequence. No model job yet.

New plan results/task_reorder/full_model_20260920/plan.json. Runtime now targets
/home/u4320956/.cache/mixfp4-model-runtime, hardlink uv cache; explicit clear only
unready venv. New sources native/model_runtime.cu; scripts/build_native_model_runtime.py,
native_model_runtime.py, check_native_model_runtime.py, benchmark_native_llama.py.
Build-local override adds true per-N256/K64 type selection to sibling CUTLASS;
no sibling edits. C ABI packed-GEMM + dynamic amax/activation quant included;
columns fused into producer, row restoration currently separate (label clearly).
Not compiled/tested yet. Full Llama harness preserves all224 projection policies,
checks source hashes and bitwise packed weight decoding, compares native logits to
fake-quant BF16 reference before matched prefill128/2048 +32 cached decode timings.
slurm/gb200_full_llama.sbatch prepared, NOT submitted. First run after kernel gate.
Qwen needs transformers5 and N48 projections support; not integrated yet.
Do not call microbenchmarks full model. Keep all prior quality gates/status intact.

# Renewed follow-up COMPLETE — 2026-09-20

All GPUjobs complete andmonitored, noactiveallocations. Llama147tile candidatepassed independent64CEgate405692; PPL405707 Wiki6.8648862839/C4 9.7969455719, improvesrawboth. Finalmaps ROOT/joint192/layouts (3gate6up10down+128background), ROOT=/work/u4320956/task_reorder/transfer_20260920/llama8b. Freshmanifest ROOT/joint192_confirm/fresh_manifest.json mustbeexcludedfuture. GB200final405810 correctlyreappliesglobalscale, all8checks pass; matchedquant+GEMM+consumer overhead1.4–5.1%, notfullmodeltiming. Pythonquantoracle405709 exact22528BF16values aftersigned-zero fix. See MIXFP4_REPORT.md, REORDERING_STUDY_STATUS.md andrenewed_job_ledger.json. Publishingcurrentresults; priorentriesbelowhistorical. User requestedgoodresultsnowachievedmodestLlamaimprovement+lowoverhead, no90%claim.

# Active update — 2026-09-20 06:25 UTC

Llama405692 NEWfresh64 PASSED fullfrozenCEgate: raw-.00167545 SE.00044778; identity-.00171237 SE.00041707; math/codebothnegative;147totaltiles,lastMLP3gate6up10down. Candidate ROOT/joint192/layouts; confirmation ROOT/joint192_confirm. PPL405707 RUNNINGH200, attachedmonitor73717 (also405709completed). Publishedcontrolsreused; onlyrefinedPPL. ROOT=/work/u4320956/task_reorder/transfer_20260920/llama8b.
GB200405668 vectorGEMM+consumer all8passed; up2048base142.21us,fused159.79us,separate319.50us.405676 packedquantstorefusion adds~0.02-0.03usdecode; preservedcodes+SM100scales. Independentoracle405695foundonlysigned-zero mismatch; correctednativecode405697+Python405709 passed22528BF16valuesbitwise.405736 fullquant+GEMM+consumer pipeline PENDING/RUNNING, attached58148, includescorrectedquantizer; commonamaxexcluded. Atmost4GPUs, actualmax2. NeedreadPPL+fullpipeline, publishreport/status/evidence, doNOTstopifPPLbad.

# Active update — 2026-09-20 06:06 UTC

405630 new8down candidate147tiles fresh64 FAILED pooledraw2SE although bothdomainCEmeansnegative; noPPL.405640 suffix192cache PENDINGnode010 until~06:15UTC (cachelocal); attached56228.405648 joint192 exact50tileflipsearch dependsafterok405640, attached62814. Fourflipsmax, freshrequired.405635 consumerfusion passed8shapesbitwise,1.8-2.7x faster thanseparateconsumer-only; summary.json.405650 vectorconsumer boundedtest pending attached67738.405644 pipeline failed downnegativecontrol due constantB; generatedsource missedjust-intime edit. FixednonconstantB NEW405651 compiling, attachedmonitorcurrentturn. ExistingacceptedQwenunchanged. DoNOTstopatfailures; userrequestscontinuationuntilgoodLlama+GB200results.

# Active update — 2026-09-20 05:56 UTC

405623 llama_refine PENDING node010; attached monitor35157. Reuses64observeddevelopment; requires changedmap, atmost2exactadd/removeflips of failed6tilecandidate. Ifpasses, NEWfresh64 via slurm/llama_fresh_candidate.sbatch tile_refine/layouts tile_refine_confirm; dynamically exclude gate_up_confirm fresh manifest.405618 fusedoutput COMPLETED all8bitwisecorrect, performancefailed (upM2048 547us vs185us separate). PreserveTMAepilogue; investigate consumerfusion.405616 developmentfailed, receipt recorded.

# Active renewed research — 2026-09-20

User explicitly requests continued Llama quality and GB200 fusion work until good
validated results. Prior accepted Qwen result and failed gates stay unchanged.
At most4 GPUs, gov113008; H200 model/cache work and GB200 native work via Slurm.
405583 suffix diagnosis COMPLETE64s: raw/full/both192bitwise audits pass; down
causes+.005368CE while gate+up gives-.0002957CE on reused64development.
Cache on node010 /tmp/u4320956/reorder_405583/tmp/llama_suffix_ywb_7mrx.
405586 first12downsubset COMPLETE31s, developmentfailed code.405613 additive
robust19subset alsofailed; actual interactions invalidate summed-single predictions.
405595 gate/up fresh64 FAILED: pooledrawCE-.00053113,2SEupper+.00001851;
codeCE+.00005790. NoPPL. Must exclude gate_up_confirm/fresh_manifest.json in
all future fresh data.405616 exactjointgreedy <=6down tiles RUNNINGnode010,
attachedmonitor8502, usescache/fit32only +remaining32 development check.
405618 GB200 fused-output prototype compiling, attachedmonitor to currentturn.
405592/405612/405614 compilediagnostics (constructor, header override, BF16return)
resolved in scripts/build_fused_permutation.py. Mainloop unchanged; build-local
CUTLASS void-D fixes disable ordinary TMAoutput; customEVT scattersBF16directly.
No fused speed/quality result yet. Existing separate-pass timings retained.
Sessionqueue previouslyfailedsqlite; attachedmonitorskeepturnactive. Continue
candidate-specific diagnosis, never repeat failed fresh candidates or loosen gate.

# Follow-up complete — 2026-09-20

Report and evidence pushed to origin/main in commit01ab5ab; remote HEAD verified. No GPU jobs remain.

Root MIXFP4_REPORT.md now includes arranging algorithm, accepted Qwen table,
negative Llama transfer, and GB200 overhead. Llama405480 fresh64 gate FAILED:
CE+0.0052064 vsraw,+0.0051475 vsidentity; math/codebothpositive; noPPL authorized
bygate, noPPLsubmitted.178tiles (lastMLP4gate/7up/39down+128rawelsewhere).
All3compactedquantizedweightsbitwiseverified. Native405441/405458 timings plus
405476 distinct-output/negativechecks complete. NoGPUjobs remain; monitors
consumed. Total1683 GPU-seconds=.4675GPUhours,maxconcurrent2. See
results/task_reorder/transfer_20260920/plan.json and REORDERING_STUDY_STATUS.md.
The historical active entries below are superseded by this completed follow-up.

# Current follow-up execution — 2026-09-20

405480 Llama fresh64 CE confirmation RUNNING (oneH200,6mincap), attachedmonitor1204. Search405455 completedall3, gate4/up7/down39; intentional boundaryguard stopped before fresh data.405477 metadata failure17s beforefresh/model, fixedcompanionreports;405480 reusesallsearches+compactions. Excludes6Qwenfreshmanifests+bothcalibrationsets+publishedC4; plan frozen before freshdata. If gate passes use prepared slurm/llama_arrange_ppl.sbatch; if fails noPPL. NativeGB200405441/405458 timings complete; strengtheneddistinct-output+negativechecks405476 complete8s, all20passed. Report draft updated. Publish finalreport/results after Llama completes.

# Active follow-up — 2026-09-20

User requested report algorithm/results, Llama transfer quality, and B200 permutation overhead using ../mixfp4. New work authorized; accepted Qwen endpoint remains unchanged. At most4 GPUs total, gov113008; H200 model work and GB200 hardware work through Slurm. Plan: results/task_reorder/transfer_20260920/plan.json. Calibration405426 COMPLETE (187 raw256 /3345 fine tiles,169.1s scoring).405455 last-MLP both SEARCH RUNNING, attachedmonitor91152. Intentional compacted/PHASE_BOUNDARY.json existence guard stops it after all3 searches before fresh loading; once boundary reached, move guard directory aside and run corrected run_llama_arrange_transfer.py, reusing all3 completed search reports. Corrected confirmation excludes6 prior Qwen fresh manifests plus both calibration sets and published C4. No fresh Llama data read yet.405441 GB200 full-gather benchmark COMPLETE133s; expensive27–204% overhead.405458 sparse output restoration COMPLETE136s, receipt consumed. Up largeM benefits (2048pipeline174.27→92.17us), down does not. Both GB200 jobs correctness passed; no fused timing/full native212tile claim. Allgov113008, <=2 GPUs actual. No new quality claims without applicable fresh gate.

# Accepted endpoint — 2026-09-20

The user explicitly accepted the measured result as good enough and asked how it
was achieved. This supersedes the earlier90%-until-success continuation demand.
No GPU jobs remain; no new experiments are authorized by the historical ACTIVE
entries below. At most4GPUs may be used concurrently for any future work.

Accepted Qwen256×64 both-axis result: Wiki7.2558336258,C4 10.1673364639,
212E0M3tiles. VersusFourOverSix:−.0312428474/−.0210285187PPL;
43.2%/54.6%of8×64gain.90%was NOT reached. Exactcompaction preserves
weights bitwise with5001movedrows/1392movedcolumns; nativeoverheadunmeasured.

404841freshconfirmation ofthe218tileFisher extension FAILED: rawCEmean
−.0016442165,SE.0011092145,2SEupper+.0005742125. Domainmeansnegative,
but frozenpooledgatefailed; noPPL andno newbestclaim. Allmonitor events consumed.
See REORDERING_STUDY_STATUS.md and published_evidence for publication snapshots.
Status,30-rowcomparison,andmeasurement snapshots were pushed toorigin/main incommit a784c1d. RemoteHEADverified. NoGPUjobs remain.

The following notes are historical research records, superseded by this endpoint.

# Reordering handoff — 2026-09-19

## Active objective — 90% recovery

User explicitly requires >=90% of published8×64 PPL improvement over FourOverSix on BOTH datasets. Qwen targets Wiki<=7.2219824791,C4<=10.1537159920. Best current43.2%/54.6%; NOT achieved. Do not finalize at a checkpoint.

**ACTIVE404841 — ce_confirm**,oneH20010mincap,attachedmonitor73686; NEWfresh64math/codeconfirmationofsinglefrozen6tileFisher-subset+bestbothcandidate. OutputEXP/fisher_subset_validate_v2_confirm. Frozen7modules45scopetiles,218totalincludingrawelsewhere. Excludesoriginalcalibration+publishedC4+first64development+all4prior64confirmationmanifests (includingfailedce_target_combinations_confirm). CE-primaryfreshgate:pooledCE2SE<0vsrawANDmatchedidentity,domainCEmeans<=0vsraw;KLdiagnostic. NO PPL untilcompletepassedgate.
404838developmentPASSEDselectedboth: rawCE-.0025468357,2SEupper-.0005289500;parentincrement-.0012842705,1SEupper-.0002411519;math/codegainsnegative. SourceEXP/fisher_subset_validate_v2. Frozen6additionaltilesacross46qkv,48qkv,48up,51o;bestbothlastMLP20tilesunchanged. No rotation or extraGEMM inthiscandidate.
If404841passes: useexisting slurm/ce_wide_ppl.sbatch ROOT EXP/fisher_subset_validate_v2_confirm EXP/fisher_subset_ppl (inspectargs); existinggate/source/dataproofauditsreusepriorcontrols. Monitornewjob. Iffails,don'trepeatfreshcandidateorPPL;recordfailureandincludeNEWmanifestinfutureexclusions. The90%targetisstillnotachieved.
Pendingoptionaluserquestion: finalmethodmustone dense256GEMM vsallowboundedsparsecorrectionpilot. Noanswerreceived; continueSINGLEdenseGEMMbydefault. NosparsecorrectionGPUjob/codewasrun. ReadonlyLlamaartifactcheck: shippedreport/mapsJSONexist butcheckpointoldpathmissing andno1x64gradientcachefound; noLlamaGPUjob/downloadsubmitted.

**COMPLETED404838 — fisher_subset_validate_v2**,node002,1H2006mincap,attachedmonitor85631. Identical6tilecandidate as404823;404823FAILEDbeforemodelrunbecausebatchfilenamewasmisspelled. Correctedv2batchbash-ncheckedandentrypointexists; sourceplanoldfailurepreserved. OutputEXP/fisher_subset_validate_v2. Anyfreshhelpercalluses source_name='fisher_subset_validate_v2'. Monitor404807completionconsumed;itsfeaturecache6tileselectvalid.

**FAILED404823 — fisher_subset_validate**, node002,1H2006mincap, attachedmonitor51726. One6tilegloballyselectedsubset + threefixedlastMLPbackgroundsraw/rows/both; 64reuseddevelopmentdocs; nofresh/PPLyet. SourceEXP/cached_fisher_subset. Genericfreshconfirmationhelpercanbeimportedandcalledwithsource_name='fisher_subset_validate' ifpassed.
**COMPLETED404807** cache-onlyglobalFisherselection:246proper<=6tilesubsets,fit-onlywinnerindices[0,1,2,3,4,6], electionCE/KL3SEnegative+domainCEmeansnegative. Dropsone51oprojand63qfrom8tilemap. NewfeaturesEXP/cached_fisher_subset/tile_features.pt (3×8×128) preservecommonprobe/cross-matrixcovariance. No newmodelpassinsearch. Attachedmonitor45352maystillawaitterminalSlurmevent;reportcompleteandreceiptalreadyrecorded.
**COMPLETED404802** all3eight-tileFisherextensionsfailedpromotion. BestbothcombinedCEvsraw-.001927112 with2SEupper-.000057200 passesrawgate; incrementalCEvsparent-.000664547 with1SEupper+.000273488 fails, mathincrement+.000629498. NoPPL/fresh.

**COMPLETED404802 — fisher_validate**, node002,1H2006mincap, attachedmonitor55138. NEW8tileFisherlayout, threefixedfinalMLPcompositionsraw/rows/both,64reuseddevelopmentdocs,existingteacherfiles. Duplicate-mapgatebeforemodelload, finitequantizedCEprimarygatebeforeNEWfreshconfirmation. EXP/fisher_validate/plan.json isfrozen.
**COMPLETED404794** 128Fisherprobes+search241.55s,8newtiles5matrices:46qkv1,48qkv1,48up3,51o2,63q1. Curvaturecache onnode002 `/tmp/u4320956/reorder_404794/tmp/fisher128_gj92jlr6`, dirs000..007each128hfiles, samecommonprobe/docacrossallmodules. AlloldCE/KLscores reusedfrom404686cache. Baseline8tilepenaltymapdifferent (46qkv2, no63q). NoPPLyet.
If404802passes: call `scripts.prepare_ce_confirmation.prepare(Path(ROOT), 'fisher_validate')` viaPythonstdlib tofreezesinglewinneranddynamicallyexcludeallpriorconfirmationmanifests; CLIchoicesneedn'tbeedited. Submitgenericce_wide_confirm.sbatch withnewdirectoryandmonitor. IfpassesindependentCEgate, genericce_wide_ppl.py supportsitslayouts; reusecontrolsnotrerun. Ifcandidatefails, doNOTsubmitPPL. Cross-matrixcurvaturejointselectionispossiblefromcachedhfields (seeRAW_CONTEXT_RESEARCH_NOTES.md),butinspectactualfinitefailurefirst.

**COMPLETED404794 — raw_context_fisher**, node002,1H2008mincap, attachedmonitor76863. ExacttinylogitFishercovarianceaudit + first2math/codeforward/timingpilot; proceed126remainingonlyifprojectedtotal<=450sec. ReusesALL128CE/KLscorecache404686; collectsoneindependentrandomFishergradient/doc, NONEWteacherorCE/KLpasses. Searches8samefixedmodules with0.5*(sumh)^2 cross-rowcurvature,64fit64electionjoint3SE,max4groups/module. LocalGaussNewtonviaSTE,notexactquantizerHessian. EXP/raw_context_fisher/report.json.
Ifnewtiles: prepared run_fisher_validate.py/slurm/fisher_validate.sbatch reuses64teachersfrompenalty_validate onnode002, exitsbeforemodelloadifexacttype-mapduplicateofraw_context_search orcached_penalty_search. NeedfreezeplanAFTERsearchcompletewithartifacthashes.3fixedlastMLPcompositionsraw/rows/both; usualdevelopment/freshCEgatebeforePPL.
**COMPLETED404793** all20individualselectedE0tileH16rotationsfailedselectionon32balanceddevdocs; nojointcandidate,nofresh/PPL. Stopthatrotationbranch. BestactualPPL43.2%/54.6%unchanged.

**COMPLETED404793 — cached_rotation_ablation**, node002, oneH200 fourmincap, attachedmonitor35670.20singletile rotations on32reuseddevelopmentdocs; selecttop1/2/4boundedsubsets byCE1SE/domainmeansvsbestboth, atmost3jointchecks onall64. MatchedFP64/BF16outputarithmeticandcandidate-specificunrotatedcontrolaudits; up to4rotatedtiles, dualactivationviews. AnywinnerneedsNEWfreshconfirmation.
**COMPLETED404773** tile-onlyall-scopes: all192unrotatedexpandedlosses bitwiseequalmatchedordinaryFP64control,128oldbaselineauditspassed. EveryrotationvariantworsenedCEvsbestboth (up+.0003428,down+.0002322,both+.0006004). NoPPL/fresh. Checksretained.

**COMPLETED 404773 — cached_tile_rotation_precise**, node002, one H200, four-minute cap, attached monitor30092. E0-tile-only H16 with dualactivationviews and shared globals, scopes up/down/both plus matchedrotatedE2controls. FP64accumulation ONLY for reference candidate/matched controls, BF16outputs; strictall64bitwiseunrotatedexpandedvsordinaryFP64loss audit. Old BF16raw/BBB replays stillbitwiseaudited. No nativekernel/speed claim; anywinnerrequiresNEWfreshvalidation beforePPL.
404762 bandH16 COMPLETE noeligiblecandidate: upCEvsbestboth+.00075006, down-.00000082, both+.00060194. NoPPL.
404770 tile-onlyBF16expandedreference FAILED22sec: arithmeticcontrol atdocument4 differedCE.00023198. No qualityconclusion. New404773 uses matchedFP64accumulation ratherthanlooseningaudit. Cache404730 reused; noteFP64forreferencearithmetic, notdeploymentformat.

**COMPLETED404730** `suffix_rotation_cache`,node002,1H2004mincap; attachedmonitor97983 / monitor_404730. Reuses64teacherfiles from completedpenalty_validate's NEWcache onnode002, buildsraw256lastMLPcaptures+original/BF16/base/E0weights+head/norm. AuditsALL64rawCE/KLagainstprior. Prepares cheaprotationexperiments, noqualitycandidateyet. Useralreadyallowedrotationafterreorder; announcednext16-elementrotationsinE0-activecolumnbands withmatchingactivationtransformandrotatedNVFP4control. Nativeoverheadmustremainunmeasureduntiltested.
404720penaltydevelopmentCOMPLETEfailedall3: bestbothvariantCE−.00118881 versusraw,2SE+.00215309;mathCE+.00028210. NoPPL/fresh. Itsnode002cache (pathinEXP/penalty_validate/report.json) retains64developmentteachers.404717was canceledpending0GPUtime dueto58minnode032queue; portable404720rebuiltteachersandrawauditsmatched.


**COMPLETED404712** `cached_penalty_search`,node002,1H2003mincap, attachedmonitor in currentturn / monitor_404712. ReusesALL128raw-context1×64scores, NOnewmodel/teacher/gradientpasses. Fixedexploratory penalty=.5*511*(tileCEdirection−tileKLdirection)^2 persequence, commonpenalty beforejoint3SE; empiricalteacher-labelgradient covariance proxy, NOTexactHessianclaim. Fourgradientassignmentrefinements perrowgroup;8columnseeds×5mixtures,<=4newgroups/module;preserveraw. OutputEXP/cached_penalty_search/report.json/layouts. Ifnewtiles, adapt run_raw_context_validate.py tothissourceanddynamiccount; exact64dev+NEWfreshCEgate beforePPL. Do notrepeatold12tilemap ifweightsidentical.
404701rawcontext12tiledevelopment COMPLETE,failedall3variants: rawCEmean−.00134339,2SEupper+.00110588; rows−.00142480,2SEupper+.00095613; both−.00219196,2SEupper+.00021471 andincrementupper+.00030101. NoPPL/fresh. Allraw/fullforwardauditspassed.
404686fullraw-context128score+searchcompleted416seconds:12tiles across6matrices,5550movedrows aftercompaction. Cache onnode002 `/tmp/u4320956/reorder_404686/tmp/raw_context128_770b7jsd` contains128scorefiles forEACH8modules,126BF16teacherfiles (0/64teacherfiles inpilotcache). Eightmodules ordered46qkv,47o,48qkv,48up,50qkv,51o,63q,63o. Pilot4046802scoresreused,STEforwardbitwiseverified;pilotcache `/tmp/u4320956/reorder_404680/tmp/raw_context_z5ab7wlp`. Preserve/reuse.
NEWconfirmationexcludesallprevioussets includingce_target_combinations_confirm. Helpersscripts/prepare_ce_confirmation.py nowacceptraw_context_validate; extendchoicefornewpenaltysourceifpassed. Final90%targetjudgedactualPPL; latestconfirmedbeststill43.2%/54.6%.
404662250tilefreshCEconfirmation FAILED:rawpooledCE−.000470028 with2SE=.002870516;mathCE+.001707019. DoNOTPPLthiscandidate orrepeatconfirmation.404666269tileextensionfailedincrementalgate:meanCEvsparent−.000169104,1SEupper+.001216895;nofresh/PPL. Bothmonitorreceiptsconsumed. LatestconfirmedPPLrecordUNCHANGED43.2%/54.6%recovery. Needbettergradientcontext/generalization, notmoreblindbandpacking. Newconfirmexclusionsmustincludece_target_combinations_confirm/fresh_manifest.json.

Critical positive-control finding404650: FIXEDknown-good8×64 also FAILS our old strict global/domainCE/KL gate on64development. Raw deltas pooledCE=-.00442728 (2SEuppernegative), KL=-.000188916 (1SEupperpositive); mathCE=-.00104587 (1SEupperpositive),mathKL=+.000063825. Gate was misaligned with user's PPL target.
New PROSPECTIVE branch, announcedto user and frozenbeforefreshdata: primaryCE pooledmean+2SE<0 vsraw, domainCEmeans<=0; finalfresh gate also vs matchedidentity. KL reportedexplicitly asdiagnostic. Per-tile directionaljointCE/KLk3 remains unchanged. No oldfailedgate is relabeledpassed. NewCEconfirmation helpers: scripts/prepare_ce_confirmation.py, run_ce_wide_confirm.py, scripts/ce_confirmation_gate.py (5boundarytests passed), run_ce_wide_ppl.py. Only run aftercompleteddevelopmentwinner; helpers excludeallpreviousconfirmationmanifests. PPL controls reusedonlyafterbitwiseaudits. Target judged by actualPPL, notdevelopmentproxy.

Completed404608hybrids: both-axisdown causesmathKLregression.404615first48packing:38newtiles24matrices,30seconds.404619all/early/late finite tests failedstrictjointgate; noPPL.40464324module16windowablation: no per-domainmeanpass, nojointtests; pooledCE/KLpromising48gate and50qkv usedinNEWcompositions404656. Allreports under ROOT/fine_rows_v2. `run_preserved_band_expand.py` prepared but NEVERsubmitted; don'tmistakeforcompletedresult.
Read REORDER_90_PERCENT_PROGRESS.md and TARGET_90_PERCENT.json. No newPPL yet; nativeB200overhead unmeasured. Continueattachedmonitorwaits and nextjustifiedexperiments untiltarget.

## New-cluster continuation (historical checkpoint)

**Current checkpoint: all jobs complete; two concrete results delivered.**
1. Independently confirmed compactedrow candidate:201total256×64tiles,
   Wiki7.262465/C4 10.172032, improvinglocalraw7.266300/10.176030.
   39933→3914movedrows, samequantizedweights. Fresh64CE/KLgatepassed.
2. Bestboth-axis model retained exactly:212tiles, Wiki7.255834/C4 10.167336.
   NEW404570 exactcompaction (23secondsH200) preservedactualquantizedweights
   whilemoving39934→5001rows(-87.5%) and27216→1392columns(-94.9%).
   Gateprojectionneedsnopermutation. PPLreusedfromidenticalparent, notremeasured.
   Frozenlayouts:`fine_rows_v2/both_exact_compaction/frozen`.
   NativeB200latencyUNMEASURED; do notinfer87.5%/94.9%runtimeimprovements.

Both-axis remains the measuredPPLleader. Its64-developmentfinitecheck404528
shows strongerCEgain butpositive mathteacherKL;20deletions andboundedsubset
prediction foundnostrictjointgatepass. ExactcompactiondoesNOTerase thattrade-off.
The6-tilewider-rowextension404506 failed: everytileworsenedmathCE. No extraPPL.
NoGPUjobisrunning orqueued; allcompletioneventshandled. The user's requestto
obtainagoodresultissupportedbybothheld-outPPLgainandtheexactlow-movement
transformationofourbestqualitymodel. Neverclaimglobaloptimalityor8×64parity.

Read `BOTH_AXIS_COMPACTION.md`, `CONFIRMED_REORDER_RESULT.md`, and the updated
30-row `results/task_reorder/cluster_20260919/FULL_COMPARISON.{md,csv,json}`.
Latestuserquestion:bothaxesarenotlesseffective—oldmatchedbothbeatsrowsinPPL;
newrowsworkwasforvalidationanddeploymentcost, notabestPPLclaim.
Next meaningful hardwarecheck is nativeB200quantization/permutation/epilogue
andwholeMLP latency withtheseexactlayouts. H200cannottestnativeB200overhead.
No rotationorLlamaexperimentswere added. Reuseexistingcontrolsandcachedata.

### Historical running status (superseded)

**Latest direction: BOTH-axis refinement404528 running onnode032,8mincap.**
Useraskedifbothaxesarelesseffective; answerNO: matchedoldboth7.255834/10.167336
beatsoldrows7.259481/10.169099 andnewconfirmedrows7.262465/10.172032.
Newrowsresultisrobust/low-movement, notbestPPL. Continuepushingbothunderidentical
finite-lossandfreshvalidationlogic.404506 rowextensionCOMPLETED:all6extra tiles
worsenmathCE;noeligiblecombinations,noextraPPL. Teacherscachedonnode032 at
`/tmp/u4320956/reorder_404506/tmp/extend_teacher_j26zfa70`.
404528 `run_both_finite_refine.py` reusesthoseteachers. All64firstdevelopmentdocs,
20singledeletionsfromoldboth20tilemap,top12removalssupport<=4095predictedsubsets,
<=3actualjointchecks,parentalsoeligible. Strictper-domain1SECE/KLvsraw+matchedid.
NoWiki/C4orfreshconfirmationdatausedforselection. Rowscanbecompactedwithcolumns
heldfixed; actualdequantizedweightsmustremainbitwiseequal. AnywinnerrequiresNEW
independentconfirmationexcludingallfourprior64sets beforePPL. Do notrerunoldPPL
ifexactparentunchanged;verifyweight/mapequivalenceandreuseitsknownPPLinstead.
Output `fine_rows_v2/both_finite_refine/report.json`; attachedcompletionmonitor
`monitor_404528/status.json`. Nextfreshscriptmustpermitbothaxes (existinggeneric
loaderdoes), expectedtilecountfromselectedmap; currentpreparehelperhardcodes12
parenttiles soADAPTthatmetadatahelperbeforeusingitfor20tilebothparent.

**PPL404483 COMPLETE (16m35s), independently confirmed9tile candidate improves both.**
Wiki7.262465000 vsraw7.266300201 (delta-0.003835201); C4 10.172032356
vsraw10.176030159 (delta-0.003997803). Both pairedNLL ±2SEintervalsbelowzero.
Matchedidentitynew7.267062664/10.176316261. 201totaltiles,9lastMLP+192rawelsewhere.
All401baseline replay logits andcandidatefirstwindow/domainfullforwardauditspassed.
Fullcomparisonupdated (`results/task_reorder/cluster_20260919/FULL_COMPARISON.*`).
This is a good low-movement validatedresult, NOT the besthistoricalPPL:
olderrows7.259481/10.169099, olderboth7.255834/10.167336 remainlowerbuttheir
freshjointCE/KLcasewasnotconfirmed. Avoidclaimingnewoverallaccuracyrecord.

**404506 stillactive**, six-tileextension, oneH200 onnode032,10mincap,
attachedmonitor57692. Finishandhandleitsresult. Ifnofurthercandidatepasses,
reporttheconfirmedPPLgain andminimaloverheadresult, noPPLfortheextension.
Ifextensionpasses, itneedsnewindependentconfirmationbeforePPL.

### Previous running status

**Active jobs:404483 PPL and404506 bounded extension (node032).**
404483 attachedmonitor78368, node25a-hgpn008, oneH200. WikiText finished:
newmatchedidentity7.267063, confirmedcompactedrow9tile7.262465 vsraw7.266300.
C4inprogress. Do notstopbeforePPLresult andextensionresult arehandled.
404506 attachedmonitor57692, oneH20010mincap, node25a-hgpn032.
404500 was cancelled pending(0GPUtime): originalcache node001fullyoccupied with
~100minutequeue andSSH accessdenied.404506 regeneratesONLY64developmentteacher
references beforequantization; layoutsandsavedbaselinelossesreuse, all64exactaudits.
`run_expand_confirmed_rows.py` tests SIXexistingextra tiles:58down1,60up1,62down4.
62up is excluded because replacingit wouldremove2rawtiles. Alltargetextra matrices
havezero rawE0M3tiles. Baseline is independently confirmedlast1nine-tilecandidate.
Replay layers58–63 withcapturedexactkwargs, raw/baseline64lossesmustmatchcachebitwise,
eachnewpolicy auditedfullforwardfirstwindow. Reuse64developmentteachers, nofreshorPPL
selectiondata. Sixsingles,<=63predictedsubsets,<=3actualjointtests, strictdomain1SE
CE/KLversusrawandconfirmedlast1. AnywinnerneedsNEWindependentconfirmationbeforePPL.
NoextraPPLhasbeenqueued. Output `fine_rows_v2/extend_confirmed_rows/report.json`.

**CONFIRMED:9-tile row-only candidate passed independent64 check404446.**
Raw ΔCE=-0.000484928 (upper1SE=-0.000305578), ΔKL=-0.0000650895
(upper1SE=-0.0000307080); both domain means improve. Matched identity also passes
pooled CE/KL1SE. Do not confuse this frozen confirmation gate with the stricter
per-domain1SE DEVELOPMENT selector: codeKL1SE crosseszero onconfirmation, butits
meanimproves as required by the predeclared confirmation protocol.

**PPL404483 RUNNING**, one H20045mincap; attached completion monitor, unique
`monitor_404483/status.json`. Out `fine_rows_v2/suffix_ce_restore_ppl`.
Only domain_pruned will be evaluated if matchedidentity masksequaloldones;
otherwise identityalso once. Base/rawcontrols reusedafterbitwise maskandexacttoken
checks. Monitor and consume completion; do not rerun controls or jobs.
Candidate9tiles within finalMLP,192rawtilesoutside =>201total256x64tiles.
Movedrows39933→3914 with bitwiseidenticalquantizedweights; nativeB200latencyunmeasured.

Matchedidentity maps differ despite equal tilecounts, so both identity andcandidate
need PPL.404476 was cancelled before PPL to avoid two fullforward passes/window.
New `run_suffix_row_eval.py` shares each fullprefix, computes both finalMLPs,
audits firstpolicy replaybitwise onALL401windows andcandidatefullforwardonfirst
windowofeachdomain. Oldattempt at `fine_rows_v2/ppl_attempts/404476`.
404483 attachedmonitorsession78368. Expected~18minutes ratherthan~36.

### Previous independent-confirmation status

**Current:404446 independently confirms frozen9-tile compacted row candidate.**
One H200,10min cap, attached monitor18797. Job404439 completed successfully using
only cached final-MLP state (~11.3GB peakGPU). All64 raw/identity replay losses
matched prior full-model results bitwise. Winner removes original tile indices
[2,6,11]; all64-development CE/KL1SE bounds pass for BOTH controls, all/math/code.
Raw deltas: CE=-0.000798200, KL=-0.000136720. This is development selection,
not independent confirmation. Actual quantized weights are bitwise unchanged by
row compaction: moved rows39933→3914(-90.2%). Native latency remains unmeasured.

Confirmation directory `fine_rows_v2/suffix_ce_restore_confirm`; excludes all
three previous64-document sets and calibration/C4 docs. If PASSED, run metadata
`scripts/frozen_confirmation_gate.py CONFIRMATION`, then submit
`slurm/validated_row_ppl.sbatch ROOT CONFIRMATION UNIQUE_OUT` and attach monitor.
New PPL wrapper verifies full raw maps against old PPL maps, and reuses matched
identity ONLY if identity masks match exactly. Fullbase/raw controls reused;
ifidentitydiffers evaluateit once alongsidecandidate. OneH20045mincap, expected
~18min ifonlycandidate. No PPL has yet been submitted.

### Previous suffix-search status

**Active job404439**, suffix-only six-candidate search, one H200 on25a-hgpn001,
4-minute cap. Attached monitor session listed in live turn; status
`results/task_reorder/cluster_20260919/monitor_404439/status.json`.
404438 was cancelled before measurements to fix mutable MLP/base buffer aliasing;
its plan is archived at `suffix_ce_restore/attempts/404438`. Source now clones
base buffers and asserts distinct storage. No fullmodel or teacher regeneration.
404434 completed3m05s: all three combined-development candidates improve pooled
CE/KL, but strict domain gate still fails (closest: identity/math CE upper+5e-6).
404439 evaluates six explicit adjacent maps using cached64 prefix inputs, teachers,
and existing32-window candidate measurements. Audits raw/identity losses bitwise
on ALL64, cached candidates on firstwindow. All64 are development selection data.
If passed: `prepare_domain_confirmation.py ROOT --source suffix_ce_restore`, then
run frozen confirmation on64 new docs with attached monitoring. No PPL beforepass.

User clarified to IGNORE unrelated messages starting with 黃寗琪; focus MixFP4 only.

### Previous expanded-development status

**Active job404434** (`domain_expand`, H200 node25a-hgpn001,5-minute cap).
Attached monitor20647, `monitor_404434/status.json`.404429 completed with no
strict32-window domain-gate pass. Several exact candidates improve all domain
means; misses include an identity-code KL upper bound of+0.00000452.
404434 adds the remaining32 development windows for THREE existing maps only,
reuses cached first32 losses and all64 teachers, applies the SAME strict gate
to64 development samples. All64 now selection data, no independent validation
claim. No new maps or relaxed bounds. If passed, freeze one using
`prepare_domain_confirmation.py ROOT --source domain_expand`, then submit
`slurm/frozen_row_confirm.sbatch ROOT EXP/domain_expand_confirm` and monitor.
Independent confirmation excludes firstfresh64 and both subsequent64 sets.

### Previous exact-refinement status

**Active job404429** (`domain_exact`, one H200 on25a-hgpn001,6-minute cap).
Attached completion monitor session36301; status `monitor_404429/status.json`.
404417 completed: three joint combinations improved pooled CE/KL and both domain
means, but none passed the prespecified stricter per-domain1SE discovery gate.
Additive predictions overstated KL improvement.404429 reuses its saved teacher
outputs and suffix inputs, directly measures seven explicit neighboring deletion
sets, retains the stricter gate and freezes one before the development check.
If it passes, use `scripts/prepare_domain_confirmation.py ROOT --source domain_exact_refine`
to freeze a compacted candidate for NEW64-window confirmation. Do not run PPL
until that new confirmation passes. Do not rerun any completed job.

### Previous refinement status

**Current status: resumed research; 404366/404376 complete, no PPL promotion.**
Finite tile deletion 404366 (4m31s) removed up_proj [44,36] and passed its
32-window original-calibration check. Independent 64-window confirmation404376
(5m27s) failed: ΔCE vs raw=-0.000733445, ΔKL=+0.000022043;
math KL=+0.000097454. This is a calibration generalization failure, not a good
confirmed result. Keep researching under the user's explicit instruction.

Running **404417**: `run_domain_tile_replay.py`, one H200 capped10min,12single deletions and
at most3joint maps. The first validation64 documents are explicitly reclassified
as development32/32; neither half can certify the resulting candidate. Exclude
both subsequent confirmation sets. Protect each domain and rank by worst-domain
KL margin rather than pooled CE. New independent confirmation remains mandatory.
Also test exact row compaction and assert dequantized weights bitwise identical.

Codex queue database corruption recurred. Attached monitor4018 detected404376's
completion and exited; its receipt is recorded. Keep this agent turn active and
start an attached completion monitor for each successor while queue is unreliable.
404417 attached monitor session11514, status `results/task_reorder/cluster_20260919/monitor_404417/status.json`.

### Earlier confirmation history (superseded status)

**Current status: confirmation 404304 completed; rejected by the frozen KL gate.**
No GPU job is running or queued. On 64 new windows, candidate-vs-raw mean
ΔCE=-0.000581664 (mean+1SE=-0.000420554); ΔKL=-0.000005558
(mean+1SE=+0.000026267). Math KL mean is +0.000058790, also failing the
predeclared domain condition. CE improved in both domains, but joint CE/KL
improvement was not confirmed. No PPL run, reranking, or threshold relaxation.
Results: `results/task_reorder/cluster_20260919/preserved_row_confirmation_summary.json`.
Confirmation took 6m39s; the diagnostic plus confirmation, including failed
allocations, used 9m30s of single-H200 allocation in total.

The completion watcher detected the finish, but queuing failed because Codex's
logging database became unreadable again. The database and WAL/SHM were preserved
with suffix `.unreadable-20260919T154940Z`; Codex recreated them and accepted the
completion notification. Watcher errors now preserve the actual CLI diagnostic;
five metadata-only monitor tests passed. This recurring logging-DB issue remains
a reliability limitation; do not claim it is permanently fixed. The completion
notification may arrive after this status update: acknowledge it without repeating jobs.

The completed confirmation history follows:
Diagnostic 404288 completed in 2m28s (2m51s including failed attempt 404285).
Its raw-preserving last-MLP row variant measured diagnostic ΔCE=-0.0006369 and
ΔKL=-0.0001637 against raw256 on 16 reused windows; raw baseline audits matched
exactly. This does not reverse the previous fresh rejection. See
[FINE_ROW_DIAGNOSIS.md](FINE_ROW_DIAGNOSIS.md) for the qualifications.

One candidate is now frozen for **64 new independent windows**, excluding prior
calibration/fresh documents and published C4 documents. It must beat raw256 and
matched last-MLP identity on CE/KL mean+1SE, with nonpositive mean differences in
both domains against raw. No search or PPL is queued. Read
`fine_rows_v2/preserved_row_confirm/report.json` after completion; monitor
`results/task_reorder/cluster_20260919/monitor_404304/status.json` notifies this session.
Submission 404303 failed before compute because pytest was missing; changed its
test runner to standard-library unittest and preserved the original plan. A
notification replay for 404303 must not cause another submission.

The prior diagnostic history follows:
The actual completion notification arrived in this session, and its receipt is
recorded in `monitor/notifications.json` and `monitor/delivery_receipts.json`.
Post-completion analysis found 17/24 matrices elected no tiles and opposing math/code
effects, including a substantial change caused by the matched identity baseline.
See [FINE_ROW_DIAGNOSIS.md](FINE_ROW_DIAGNOSIS.md). Diagnostic **404285** restores
raw maps only in inactive layers of two already-frozen row panels: one H200,
16 reused development windows, hard 10-minute limit, no search or PPL. Its
notification-capable monitor is `monitor_404285/status.json`. Upon completion,
read `fine_rows_v2/inactive_raw_ablation/report.json`; no automatic successor is queued.
Attempt 404285 failed in 23 seconds on 25a-hgpn003: CUDA could not see its GPU,
before any evaluation sequence. Its report/plan are preserved under
`inactive_raw_ablation/attempts/404285/`. Retry **404288** excludes that node and
checks CUDA before loading weights. Its hard limit is 9m30s, keeping total
allocated time under ten minutes including the first attempt. No third attempt
without new evidence. Current completion monitor: `monitor_404288/status.json`.

All 403637 array
tasks and fresh validation 403638 completed successfully. Both `selected_any` and
`selected_rows` fell back to raw256: no candidate passed the prespecified fresh
CE/KL gate. The metadata gate returned no PPL policies, so no successor GPU job
was submitted. Results and source hash:
`results/task_reorder/cluster_20260919/fine_rows_v2_summary.json`.

**Always monitor completion, not only log files.** `AGENTS.md` records the user's
requirement. The watcher now supports `--notify-thread "$CODEX_THREAD_ID"` and
queues terminal-state notifications through `codex queue`. Five metadata-only
tests passed. A real completion message was accepted for this session; its queue
receipt is in `monitor/notifications.json`. Agent receipt must be acknowledged
when the queued message arrives, not inferred from queue acceptance. All watched
jobs are terminal, so this watcher has exited normally. Start notification-capable
watchers for any new jobs. If notification fails, keep an attached monitor active.
The connectivity-test message subsequently arrived in this same session after
the previous turn ended, verifying the wake-up path. Its acknowledgement is in
`monitor/delivery_receipts.json`. The separate batch completion notification has
also arrived and been acknowledged; completion notifications do not expand scope.
The first queue test failed because `~/.codex/logs_2.sqlite` lacked a SQLite header;
the original was preserved as `logs_2.sqlite.invalid-header-20260919T151918Z`, and
Codex recreated its logging database. No research data or session history was removed.

The following launch/resource notes describe the now-completed batch:
[FINE_ROW_RESEARCH.md](FINE_ROW_RESEARCH.md).
The user requested continued research. Fine individual-row (1×64 atom) search
now runs across layers 56–63 with deployment-matched activation factors and
background weights. Full local 8×64/256×64 controls have 3,787/195 tiles.
Jobs: 403636 (controls completed), 403637_[0-7] (eight score/search workers),
403638 (fresh 64-sequence CE/KL selection). **403639 cancelled while pending**
after the user requested economical GPU use. No PPL job is currently queued.
After fresh validation, run `scripts/fine_row_eval_policy.py ROOT` on the login
node: empty output means stop; otherwise the revised one-GPU evaluator measures
only the selected row-first winner and matched identity. Reuse completed full
controls. Do not launch more layer sweeps or Llama calibration without a concrete
reason supported by this batch. Current CPU searches retain eight GPU reservations;
future CPU search must consolidate allocations or use a CPU-capable compute queue.
The dev partition rejected a CPU-only scheduling dry run. No heavy login work.

Full local controls: raw256 **7.266300 / 10.176030**, full 8×64 **7.212709 /
10.150963**, FourOverSix **7.287076 / 10.188365**. HF credential is stored outside
the repo (0600); authenticated pinned Llama access succeeded, but no Llama download
or compute was launched. Compact-mask calibration support passed tests but is unrun.
Tests 403685 passed 25/25; real Qwen smoke 403635 passed. Preparation 403618 crashed during
Python shutdown after saving its outputs; 403628 independently verified every
saved control and fresh tensor before the chain was restarted. The canonical
60-second watcher now tracks this batch. No heavy work runs on the login node.

The completed-batch descriptions below are retained as history.

**Final batch completed:** all 402766 array tasks and four-GPU evaluation
402786 finished. The wider row-band policy measured 7.263998 / 10.175365;
its paired intervals versus local raw256 cross zero. No jobs remain running
from this experiment. The comparison now has 27 rows, including both models'
reference results and all distinct pilot policies, is in
[FULL_COMPARISON.md](results/task_reorder/cluster_20260919/FULL_COMPARISON.md)
and [CSV](results/task_reorder/cluster_20260919/FULL_COMPARISON.csv).
The job scheduling description below records how the completed batch ran.

**Latest results:** [REORDER_RESULTS_20260919.md](REORDER_RESULTS_20260919.md).
Jobs 402294, 402295, 402309, 402322 and 402323 all completed successfully.
With the local raw-256 background, both-axis last-MLP reordering measured
**7.255834 WikiText / 10.167336 C4**; row-only **7.259481 / 10.169099**.
The matched identity control is 7.266998 / 10.176431; both reorders have negative
paired ΔNLL ±2SE intervals on both datasets. The fully foldable shared search
did not improve quality. Aggressive map `rows_k1` won inner finite-loss selection
but worsened teacher KL on separate outer confirmation and was rejected.

**Completed final batch:** 402766_[0-7] searched row-only eight-row group permutations
across layers 56–63 (24 MLP matrices); **402786** evaluates four matched panels
after all searches using **four H200 GPUs**, two policies per GPU. It replaces
cancelled single-GPU job 402769. The user requested more concurrency: all
**eight** H200 search workers now run concurrently, each covering three matrices
(the search itself is CPU computation on compute nodes). This reuses historical
scores without duplicating persistent data. Tests 402785 passed **21/21**.
The restarted watcher tracks the array, new evaluation job, and all four GPU logs;
it polls every 60 seconds and logs status/errors but does not repair failures.

FourOverSix reproduces exactly: **7.287076 / 10.188365**. The local full raw-256
map is **195 tiles, 7.266300 / 10.176030**, versus the user's **198 tiles,
7.275704 / 10.177685**. These are distinct baselines; the precise cause of map
differences remains unresolved. Use matched local controls for reorder gains.

The recovery-stage job table below is retained as history.

The sections below this update describe the **previous cluster**. Its job IDs
are not meaningful here: local accounting reuses the same numbers for unrelated
jobs, and its score files and Python environment were not transferred.

Current login host: `25a-lgn01`. `wallet` confirms `GOV113008`,
`TAIDE計畫POC_中研院黃瀚萱教授`. All five dedicated `taide` H200 nodes were
`DOWN+NOT_RESPONDING` at submission. The jobs below therefore use the **same
gov113008 account on the available H200 `dev` partition**, with its four-hour
limit. No calibration, search, model evaluation or tests ran on the login node.

The user reports verified equal GEMM speed for 256x64 MixFP4 and NVFP4.
Reordering overhead is still to be measured. Rotation is deferred.

### Current jobs

**Recovery update:** job `402043` finished all 128 scoring sequences but failed
exporting historical shard `238.pt`: the user's `/work` quota is **100 GiB**,
despite `df /work` initially showing the whole filesystem capacity. Some final
historical shards are truncated/empty. All 384 fine pilot shards and their
three manifests had been written before the failure. The original dependent
jobs `402046`, `402051`, and `402061` were automatically cancelled.

Recovery job `402172` moves this experiment's model cache to
`.joblib/huggingface` on `/home` (also a 100 GiB user quota), then loads, checks
finiteness/shape/sequence identity, and hashes every pilot shard. It preserves
the failed report and writes a separate `pilot_calibration/report.json` with
status `pilot_complete`, explicitly **not** a completed full-model calibration.
The cache move freed about 52 GiB on `/work`; `/home` has about 40 GiB remaining.

The full calibration retry retains the validated pilot files unchanged,
recomputes historical scores, and replaces incomplete exports atomically after
reading them back and checking equality. Output quota is checked before compute.
It records which earlier job supplied the reused pilot. The isolated pilot
evaluation can proceed independently while the full-model control is repaired.

| Stage | Job | State at handoff update |
|---|---|---|
| Pinned Python environments + original tests | 402040 | Completed; torch 2.9.0+cu128, Transformers 4.57.3 / 5.16.1; 9/9 tests |
| Expanded tests + syntax/shell checks | 402173 | Completed; 12/12 tests |
| Cache relocation + pilot validation | 402172 | Completed in 2m47s; all 384 paired shards valid |
| Full-model calibration retry, preserving pilot | 402175 | Running; 2 H200, 24 CPUs, 400G |
| Qwen searches, three modules × both/rows/cols | 402174_[0-8] | Tasks 0/1 running; max two concurrent |
| Qwen matched pilot quality evaluation | 402177 | Depends on all nine searches; uses pilot_calibration snapshot |
| Qwen full raw-256 background evaluation | 402178 | Depends on searches and full calibration retry |

Cancelled setup job `402038` was superseded after finding the taide nodes down.
Test job `402049` exposed an incomplete scale-group validator; it was fixed and
`402050` passed. No existing jobs from other experiments were changed.
Job `402059` also passed 11/11 after adding search progress logging; `402060`
includes the full-model coarse-control covariance/tail test and passed 12/12.

Qwen artifacts: `/work/u4320956/task_reorder/pilot_20260919/qwen27b/`.
The six evaluation policies are FourOverSix, identity 256x64, identity 8x64,
joint reorder, row-only reorder and column-only reorder. Only the three final
MLP projections switch types; everything else stays FourOverSix. The 8x64
control uses the same independent election half as the coarse controls, so
it is not the full-model 128-sequence row in the root report.

The user supplied full-model raw-256 results during this continuation, saved
with explicit provenance in
`results/task_reorder/cluster_20260919/raw256_reference.json`:

| Model | Raw 256x64 tiles | WikiText-2 | C4 |
|---|---:|---:|---:|
| Qwen3.8-27B | 198 | 7.275704 | 10.177685 |
| Llama-3.1-8B | 187 | 6.866879 | 9.801361 |

Job `402178` writes `evaluation_raw256/`: it re-elects the full-model raw map
from all 128 historical score sequences (aggregate each sequence to N256 before
computing mean+3SE), saves that map, and evaluates FourOverSix and raw256 as
complete-model references in the same job. It then fixes all non-pilot weights
at that raw map while comparing the matched pilot identity/reordered maps.
Pilot elections remain on the independent 64-sequence half. Both comparisons
are recorded: against the full raw reference, and against the matched pilot
identity control. The latter isolates reordering from the split change. The
report records numerical deviations from the user-supplied full-model values;
it does not silently assume cross-cluster reproduction.

Llama remains waiting for a local checkpoint path or Hugging Face credential
path. The previous `/work/u4320956/hf/...` checkpoint is absent, and no HF_TOKEN
or default HF token file was available in this session. The user was asked for
a path, not a token pasted into chat. The calibration runner now accepts
`--model-source PATH_OR_HUB_ID` while preserving revision and every weight hash.

### Added in this continuation

- Worker environment setup and H200 scripts; cached downloads are shared across
  stages in `.joblib/huggingface` on `/home`, scores on `/work`, and teacher
  logits in job-local temporary storage. Both shared user quotas are 100 GiB.
- Frozen identity 256x64 and 8x64 masks exported with every searched layout.
- `run_task_reorder_eval.py`: matched control checks, exact source weight checks,
  frozen layout hashes, historical-map comparison, published evaluation protocol,
  and paired per-window NLL differences. No cross-GPU bitwise reproduction claim.
- `original_order_weight_reference`: a tested quality reference that undoes
  deployment permutations and rejects split scale groups.
- Array submission for the nine searches and dependency-linked evaluation.
- Configurable log stems and array IDs in `scripts/watch_mixfp4_jobs.py`.
- Per-layout metadata for permutation index storage and unfused memory traffic.

Monitor scheduler/log status in
`results/task_reorder/cluster_20260919/monitor/status.json` and bounded log tails:

```bash
squeue -a -u u4320956
sacct -j 402172,402175,402174,402177,402178 --format=JobID,State,Elapsed,ExitCode
tail -c 4000 slurm/logs/reorder_cal_402175.out
tail -c 4000 slurm/logs/reorder_cal_402175.err
```

Require a complete calibration or the separately validated `pilot_complete`
snapshot and all three complete manifests before interpreting pilot scores.
The full-model control still requires a complete full calibration. Require successful search artifacts and completed matched
evaluation before claiming quality gains. Search objective improvement alone
is not a quality result. Continue from the resulting row/column comparison;
prefer the cheaper deployment if its held-out quality is comparable. Keep the
root 8x64 report unchanged until new completed measurements justify an update.

---

## Previous-cluster handoff (historical)

Continue with **Llama-3.1-8B and Qwen3.8-27B only**; skip Qwen3-4B.
Target `m256n256k64`, hence weight type tiles **N256 x K64**.
Only weights mix E2M1/E0M3; activations retain FourOverSix E2M1.
All compute, including CPU search and tests, goes through Slurm under
`gov113008`, using `taide` (H100) or `taide_h200` (H200).

## Implemented and checked

- [Plan](MIXFP4_GB200_PLAN.md), linked from the root MixFP4 report and its generator.
- [Method and artifact contract](TASK_REORDER.md): joint row/column search,
  capacity-constrained assignments, exact improving swaps, multistart search,
  fit/election split, and legal 16-column scale-group permutations.
- `run_math_code_calibration.py --reorder-modules REGEX` streams per-sequence
  CE/KL scores at 1x16 granularity. Historical 8x64 scores cannot replace these.
- `run_task_reorder.py` searches one matrix and exports its permutations and
  independently elected 256x64 map.
- Nine focused tests passed in worker step `348924.0`, then again in job
  `349179` before calibration (9/9, 0.629 seconds). Worker syntax checks passed
  in step `348924.1`.

No real-model reordering quality or GB200 throughput result exists yet.
Rotation and the native fused kernel remain planned work.

## Submitted calibration jobs

Both jobs were submitted from `/home/u4320956/NVFP4-RaZeR` on this cluster.
They continue independently of this chat or client machine.

| Model | Job | Resources | Output |
|---|---|---|---|
| Llama-3.1-8B | 349178 | taide, 1 H100, 4 CPUs, 96G | `results/task_reorder/pilot_20260919/llama8b/calibration` |
| Qwen3.8-27B | 349179 | taide_h200, 2 H200, 24 CPUs, 400G | `results/task_reorder/pilot_20260919/qwen27b/calibration` |

At the last check, Llama was pending and Qwen was running, downloading model
files after passing the tests. Check the scheduler for current state:

```bash
squeue -a -u u4320956
sacct -j 349178,349179 --format=JobID,State,Elapsed,ExitCode
tail -c 8000 slurm/logs/reorder_cal_349179.out
tail -c 4000 slurm/logs/reorder_cal_349179.err
```

Use `-a` with `squeue`: the taide partitions are hidden from the default view.
Read bounded byte tails; model download progress uses carriage returns and
can make `tail -n` unexpectedly huge. Poll no more frequently than 30 seconds.

The pilot covers the final layer's **gate_proj, up_proj, down_proj**:
`model.layers.31.mlp.*` for Llama and
`model.language_model.layers.63.mlp.*` for Qwen.
The collector still computes the historical full-model scores as well.
Calibration uses 128 pinned math/code sequences. Search uses 64; independent
election uses the other 64, stratified by source.

The batch script uses `--allow-source-drift` because commit `384b803` appended
the unused paired FourOverSix helper to `quantize/quantizer.py`; the existing
quantizer functions were unchanged. The calibration report records this drift.
Check regenerated historical maps against the shipped calibration maps before
claiming reproduction. Transformer versions are 4.57.3 (Llama) and 5.16.1
(Qwen); job-local dependencies and HF downloads live under worker `/tmp`.

## Work still to do

**Only calibration has been submitted. Search/evaluation dependencies have
not been wired, and no new-job background monitor was started before handoff.**
`scripts/watch_mixfp4_jobs.py` currently knows the earlier coarsening job log
prefixes; generalize those paths before using it for these new jobs.

1. Monitor the two calibration jobs. Require a complete `report.json` and
   all three complete `reorder_scores/NNN/manifest.json` files. Preserve scores
   on shared storage: `.pt` files and Slurm logs are intentionally excluded
   from Git and do not arrive with a clone on another machine.
2. Submit one search per module using the supplied worker script, e.g. after
   successful Llama calibration:

   ```bash
   sbatch --dependency=afterok:349178 --kill-on-invalid-dep=yes \
     slurm/task_reorder.sbatch \
     --scores results/task_reorder/pilot_20260919/llama8b/calibration/reorder_scores/000 \
     --out results/task_reorder/pilot_20260919/llama8b/search/000
   ```

   Repeat for `001` and `002`; Qwen uses dependency `349179` and model directory
   `qwen27b`. Consult each manifest for the module-to-index mapping. Defaults
   are k=3, seed=0, four starts, six rounds, both axes, and 256x64 tiles.
3. Implement the matched model-quality evaluator. Compare FourOverSix,
   identity-layout 256x64, and reordered 256x64 on **the same pilot modules**;
   keep all other weights at FourOverSix. Freeze layouts before evaluating.
   `run_task_reorder.py` currently reports identity election counts/objective
   but does not save the identity mask: export that mask or regenerate it
   from the identical election split for the matched control.
4. Reuse the root report's protocol (`run_kse_paper.py` and
   `run_baseline_protocol_audit.data`): 2048 tokens, 141 WikiText windows,
   256 seed-0 C4 crops, tensor-wide activation factors, SDPA, WikiText cached,
   C4 uncached, float32 PPL aggregation, and paired per-window NLL differences.
   Calibration uses causal per-token activation factors, as recorded in the
   manifests; evaluation must retain the published tensor-wide convention.
5. The exported reference weights are in permuted deployment order. A fake
   quantization quality evaluator may undo both permutations on those mixed
   weights to run the original model graph, preserving each 16-element scale
   group. Validate that algebra separately from native runtime performance.
   No speed claim follows from a quality simulation.

The existing coarsening experiments in `.claude/worktrees/tile256` are separate
work. Do not modify or cancel their jobs while continuing this pilot.
