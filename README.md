# Spatial-Aware Multi-Person Re-Identification

<!-- Lead with the current output: a sample, then the demo. Write once the project is done. -->

```mermaid
flowchart LR
    C["Current frame<br/>people, labeled"] --> K[("Bank<br/>everyone seen so far")]
    N["Next frame"] --> D["Detector"]
    D -- boxes --> E["Embedder"]
    E -- a vector per box --> M["Association"]
    K -- candidates --> M
    M --> L["Next frame<br/>people, labeled"]
    M -- updates --> K
```

*The tracker, one frame at a time.*

This project builds on a presentation my groupmates, Mark Gallardo and Caine Bautista, and I delivered at the 8th Collaborative Online International Learning (COIL) Conference. Our approach at the time scored each pair of people, one from the current frame and one from the next, in isolation. When following one person, the best-scoring match decided which person in the next frame, if any, was the tracked one. When tracking everyone, the scores went through Hungarian matching[^kuhn]. Attention mixed three inputs per pair: a learnable summary token, the difference between the two people's appearance embeddings, and their element-wise product. The summary token's output was combined with spatial and temporal features (position offsets, overlap, scale, time gap) to produce a match score.

Scoring pairs in isolation has a blind spot: **one match can't inform another.** So I wanted to match everyone at once, with the spatial features from the original project guiding it.

Trackers also tend to lose people who stay hidden for more than a moment. In the training videos of MOT17, the 2017 Multiple Object Tracking benchmark[^mot16], 47 of the 587 occlusions last longer than two seconds[^stats], twice ByteTrack's default wait of one second[^bytetrack]. So when a person loses their detection, whether behind an obstacle or off-screen, they go into a **bank** that remembers their appearance and last known state, and decides how long to keep them before calling them gone. The same bank lets the tracker re-identify people who leave the frame and come back.

Every result below is on the MOT17 validation half (Phase 0 explains the split), with the same public detections for every tracker, unless a table says otherwise. Those detections come from the Faster Region-based Convolutional Neural Network (Faster R-CNN)[^fasterrcnn], labeled FRCNN in MOT17's files.

## Phase 0: The dataset

### Where it comes from

I use [MOT17](https://motchallenge.net/data/MOT17/)[^mot16], the standard benchmark for tracking pedestrians. It has 14 short videos of pedestrian scenes, some from static cameras and some filmed while walking or driving. Seven are for training and come with ground truth: every person is boxed in every frame, with an identity number (ID) and a visibility score from 0 to 1 for how much of them is showing. The other seven are the test set, with no public ground truth.

### What it looked like, and what I fixed

Before running a single evaluation, I went through the release and found seven problems:

| Found | Fixed |
|---|---|
| Every video shipped three times, once per public detector: Deformable Part Models (DPM)[^dpm], FRCNN, and Scale-Dependent Pooling (SDP)[^sdp]. Images and ground truth are identical across the copies | Verified all 14 by content hash and collapsed them into one copy, with the three detection files side by side. That saves 3.9 GB and closes a quiet leak: split by detector folder, and the same video lands in both training and validation |
| Detection files out of frame order, including 246 rows in MOT17-02's FRCNN file | Sorted on load |
| DPM uses ten columns and unbounded scores (−0.5 to 4.77); FRCNN and SDP use seven, with scores between 0 and 1 | One parser for all three, with a score threshold per detector |
| Boxes as 1-based `left, top, width, height` | 0-based corners internally, converted back when writing results |
| 14.6% of pedestrian boxes, about one in seven, extend past the image border | Clipped before cropping |
| Static people, people on vehicles, and reflections labeled next to pedestrians | Only pedestrians marked for evaluation count as targets, following the benchmark rules[^mot16] |
| People stay annotated while fully hidden, so their crops show whatever is in front of them. Nearly one training crop in five (18.9%) has visibility below 0.2 | Kept out of the appearance model's training data |

Last, I cut each training video in time: the first half for training, the second for validation, which I'll call the training half and the validation half. This is CenterTrack's convention[^centertrack], which keeps the results comparable with published work. The test videos stay untouched.

### What it looks like now

```
data/mot17/
├── manifest.json       per video: frame rate, size, moving camera or not, split boundaries
├── train/MOT17-02/     one copy of each of the 7 training videos
│   ├── img1/           frames
│   ├── det/            DPM.txt, FRCNN.txt, SDP.txt
│   ├── gt/gt.txt
│   └── seqinfo.ini
├── test/               the 7 test videos, same layout, no ground truth
└── trackeval/          ground truth per split, in the format the scorer reads
```

In the table below, FPS is frames per second. A *gap* is a stretch where a person's visibility drops below 0.1 and later recovers, and "mostly hidden" is the share of boxes with visibility below 0.25.

| Sequence | FPS | Frames | Size | Camera | People | Boxes | Mostly hidden | Past border | Gaps | > 1 s | > 2 s | Longest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT17-02 | 30 | 600 | 1920×1080 | static | 62 | 18,581 | 49.4% | 1.7% | 144 | 55 | 24 | 11.0 s |
| MOT17-04 | 30 | 1050 | 1920×1080 | static | 83 | 47,557 | 16.3% | 26.0% | 43 | 11 | 7 | 8.9 s |
| MOT17-05 | 14 | 837 | 640×480 | moving | 133 | 6,917 | 31.1% | 22.4% | 112 | 12 | 3 | 8.6 s |
| MOT17-09 | 30 | 525 | 1920×1080 | static | 26 | 5,325 | 29.7% | 11.1% | 41 | 12 | 5 | 7.0 s |
| MOT17-10 | 30 | 654 | 1920×1080 | moving | 57 | 12,839 | 17.8% | 3.9% | 110 | 12 | 4 | 2.8 s |
| MOT17-11 | 30 | 900 | 1920×1080 | moving | 75 | 9,436 | 20.3% | 7.2% | 41 | 7 | 4 | 2.8 s |
| MOT17-13 | 25 | 750 | 1920×1080 | moving | 110 | 11,642 | 13.9% | 3.5% | 96 | 1 | 0 | 1.4 s |
| **Total** | | **5,316** | | | **546** | **112,297** | **23.6%** | **14.6%** | **587** | **110** | **47** | **11.0 s** |

Three numbers shaped everything after this:

- **Nearly a quarter of all boxes are mostly hidden**, and on MOT17-02 it's half.
- **Frame rates range from 14 to 30 FPS**, so every time limit is in seconds, not frames.
- **Four of the seven cameras move.**

One more is worth knowing: **152 of the 339 people in the validation half, 45%, also appear in the training half.** An appearance model could simply memorize them, and Phase 2 has to account for that.

## Phase 1: Baselines

### Why a baseline

A tracking score means little on its own. The same tracker scores differently with different detections, splits, and scorers, and published results mostly come from each paper's own detector, so they can't be lined up against mine. So I built my own reference points: three standard trackers, run under exactly the conditions my tracker will face.

- **SORT**[^sort], short for Simple Online and Realtime Tracking, matches people by motion only.
- **ByteTrack**[^bytetrack] adds a second pass that recovers low-confidence detections.
- **DeepSORT**[^deepsort] adds an appearance memory. Here it uses the Omni-Scale Network (OSNet)[^osnet], trained on the Multi-Scene Multi-Time person dataset (MSMT17)[^msmt17]. It looks up the closest-looking person and stores every sighting, which makes it the simplest version of the bank.

### Running them under the same conditions

- **Same detections.** Every tracker gets the public FRCNN boxes, so detection quality is fixed and any difference comes from association, meaning deciding which box belongs to whom.
- **Tuned on one half, scored on the other.** Every threshold was tuned on the training half and reported on the validation half, so no number was tuned on the data it's scored on.
- **Same code path.** I reimplemented all three from their papers, so they share one data loader, one runner, and one scorer. That also keeps this repository Apache-2.0, since the original SORT and DeepSORT code is under the GNU General Public License (GPL).

**A scorer I checked first.** Scoring uses TrackEval[^hota], which reports these:

| Metric | Stands for | Measures |
|---|---|---|
| HOTA | Higher Order Tracking Accuracy[^hota] | detection and association together, by combining the next two |
| DetA | Detection Accuracy | how well people are found and placed |
| AssA | Association Accuracy | how well each person keeps one ID over time |
| MOTA | Multiple Object Tracking Accuracy[^clear] | misses, false alarms, and ID switches in one number, dominated by detection |
| IDF1 | Identification F1 score[^idf1] | how many boxes carry the right ID |
| ID switches | | how often a person's ID changes; the only one where lower is better |

A perfect score only means something if a bad tracker can't get one. So I scored two extremes: the cleaned ground truth as if it were tracker output, and the raw detections with no tracking at all, a new ID for every box.

| Output | HOTA | DetA | AssA | MOTA | IDF1 | ID switches |
|---|---|---|---|---|---|---|
| Ground truth | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0 |
| FRCNN detections, no tracking | 5.0 | 43.6 | 0.6 | −0.7 | 0.7 | 26,306 |

The ground truth gets a perfect 100. The untracked detections keep a DetA of 43.6 while AssA falls to **0.6**: the scorer punishes lost identities without touching detection.

The baselines on the validation half:

| Tracker | HOTA | AssA | IDF1 | ID switches | Matching time |
|---|---|---|---|---|---|
| SORT | 48.4 | 56.2 | 54.5 | 222 | 0.53 ms/frame |
| ByteTrack | 49.4 | 57.7 | 56.2 | 198 | 0.79 ms/frame |
| DeepSORT + OSNet | **51.3** | **62.6** | **59.9** | **121** | 1.83 ms/frame |

DeepSORT makes **39% fewer ID switches** than ByteTrack, 121 against 198, and scores 1.9 HOTA higher. **It's the bar to clear.**

### How my tracker will be measured against them

- **HOTA first**, since it's the one number that covers both detection and association. With the detections fixed, DetA should barely move, so the differences will show up in **AssA**, **IDF1**, and **ID switches**.
- **Same features.** Whenever the appearance model changes, DeepSORT gets the same one, so a win can't come from better features alone.
- **Every part against its plain version.** Each part of my tracker has a simpler counterpart in DeepSORT, and it has to beat that counterpart on the validation half to stay.
- **Repeated runs.** Anything learned is trained more than once and reported as mean ± standard deviation.
- **Time.** The budget is 33 ms per frame, 30 frames per second, on my laptop's RTX 4050, so matching time is reported next to accuracy.

## Phase 2: The appearance model

### Where it fits

```mermaid
flowchart LR
    C["Current frame<br/>people, labeled"] --> K[("Bank<br/>everyone seen so far")]
    N["Next frame"] --> D["Detector"]
    D -- boxes --> E["<b>Embedder</b>"]
    E -- a vector per box --> M["Association"]
    K -- candidates --> M
    M --> L["Next frame<br/>people, labeled"]
    M -- updates --> K
    classDef focus stroke-width:3px,font-weight:bold
    class E focus
```

The embedder is the appearance model this phase trains. It crops each detected box out of the frame and turns it into a vector of 512 numbers. For now the detector is MOT17's public FRCNN boxes, and the bank and association come later.

### What it does

The COIL version compared people inside a model: attention over each pair's embedding difference and product. That's one model run per pair, and the number of pairs grows with the square of the crowd. So this phase moves the comparing into training instead. **The appearance model turns each crop into a vector once, and comparing two people is a single dot product**, their cosine similarity. It's trained so crops of the same person land close together and different people land far apart.

| | Per frame, with N new boxes and M known people |
|---|---|
| A model on every pair | N × M model runs |
| An embedding per crop | N model runs, then N × M dot products |

On MOT17-04, with about 27 people in every frame, that's roughly 729 model runs against 27, 27× fewer.

### How it's trained

**The candidates.** I trained three: OSNet x1.0, the full-width model, as the accuracy reference, and two faster ones, OSNet x0.5 (half the channels) and ResNet18, an 18-layer residual network[^resnet]. I timed every candidate before training any of them, and ResNet18 surprised me. It does twice the arithmetic per crop of OSNet x1.0 and still ran faster at every batch size I tried, because the graphics processor (GPU) is built for its dense convolutions.

**The crops.** I cut a crop of every visible pedestrian (visibility at least 0.3) in the training half, with the same GPU cropping the tracker uses, so training and tracking see identical pixels. Scrolling through them turned up crops that were just legs. The cause: MOT17's visibility score counts occlusion by other people but ignores the image border, so someone half outside the frame can still read as fully visible. Dropping anyone less than 60% inside the frame removed 1,290 crops, 3.0% of them, and left **41,978 crops of 325 people**.

**The recipe.** Training follows *Bag of Tricks*[^bagoftricks]: an identity classifier with label smoothing, plus a triplet loss[^triplet] that pulls each person's least similar crop closer than their most similar stranger. I changed two things for tracking:

- **Every batch comes from one video.** Sixteen people from the same sequence share its lighting and background, which are exactly the look-alikes the tracker has to tell apart.
- **Each person's four crops come from four different stretches of their track.** Matching across time is harder than matching neighboring frames, and it's what the bank needs.

Augmentations cover the camera problems I wanted the model to shrug off: brightness, contrast, saturation, and warmth changes, a lighting change across part of the crop, random erasing[^erasing] for partial blocking, plus flips and small shifts.

**The score.** I cut 11,555 crops of 305 people from every third frame of the validation half. Only people the model has never seen are used as queries, since 45% of validation people also appear in training (Phase 0). Each query searches for the same person among everyone else in its video, and I report mean average precision (mAP), meaning how high the right person's crops rank on average, and Rank-1, meaning how often the top result is the right person. Crops of the same person within a second of the query don't count, so near-identical neighboring frames can't inflate the score. Everything downstream uses the final checkpoint, not the best-scoring one, because picking by validation score would leak the validation half into the choice.

| Model | Starts from | Parameters | mAP | Rank-1 |
|---|---|---|---|---|
| OSNet x1.0, off the shelf | MSMT17 | 2.2 M | 71.2 | 80.8 |
| OSNet x1.0, fine-tuned | MSMT17 | 2.2 M | 77.9 | **88.1** |
| OSNet x0.5, fine-tuned | MSMT17 | 0.6 M | **78.1** | **88.1** |
| ResNet18, fine-tuned | ImageNet | 11.2 M | 75.5 | 86.6 |

Fine-tuning added **6.7 mAP** and **7.3 points of Rank-1** to OSNet x1.0 on people it never saw. OSNet x0.5 matched it with **a quarter of the parameters**, 78.1 against 77.9, a gap smaller than its own score moved over the last five epochs (77.5 to 78.1). ResNet18 finished 2.4 mAP behind, starting from ImageNet rather than a person dataset.

### Choosing the model

mAP only says how well a model finds people. What the tracker needs is how well it tracks and how long it takes. So I ran DeepSORT on the validation half with each model's features, and timed each model cropping and embedding every detection on the real validation frames.

Each model gets its own DeepSORT cutoff, set without labels. Two detections in the same frame are always different people, so the cutoff is whatever lets through the same share of same-frame pairs, 0.118%, as DeepSORT's tuned 0.2 did with the off-the-shelf features.

| Model | Cutoff | HOTA | AssA | IDF1 | ID switches | Embed, plain | Embed, graphed |
|---|---|---|---|---|---|---|---|
| OSNet x1.0 | 0.368 | **51.9** | **63.8** | **60.8** | **98** | 15.0 ms | 9.5 ms |
| OSNet x0.5 | 0.375 | **51.9** | 63.7 | 60.6 | 102 | 13.9 ms | 5.4 ms |
| ResNet18 | 0.312 | 51.5 | 62.8 | 59.9 | 106 | **5.2 ms** | 5.9 ms |

(Embedding time is per validation frame, averaged over all seven videos, cropping included. *Graphed* means replayed as a CUDA graph: NVIDIA's Compute Unified Device Architecture records the model's GPU work once and replays it as a single launch.)

**OSNet x0.5 it is.** It tracks as well as OSNet x1.0, 51.9 HOTA each, at 5.4 ms against 9.5 ms, **1.8× faster**. ResNet18 is 0.2 ms faster still, but gives up 0.4 HOTA, 0.9 AssA, and 4 more ID switches against x0.5.

The graphs are doing more work than it looks. Without them, x0.5 takes 13.9 ms, because at these batch sizes its time goes into launching its many small layers, not into the math. ResNet18 gets nothing from them, 5.2 ms plain against 5.9 ms graphed, since padding each frame's crops up to a fixed batch size costs more than the launches it saves.

Each model was trained once, and 0.4 HOTA is a small gap. The choice doesn't hinge on it, though: x0.5 and x1.0 tie on accuracy, and at equal accuracy the faster model wins.

### Cross-fitting

```mermaid
flowchart LR
    subgraph T["Training half"]
        A["Fold A videos<br/>MOT17-04, 05, 11"]
        B["Fold B videos<br/>MOT17-02, 09, 10, 13"]
    end
    A -- trains --> MA["Model A"]
    B -- trains --> MB["Model B"]
    B -. embedded by .-> MA
    A -. embedded by .-> MB
    MA --> X["Cross-fit features<br/>all 7 videos, every person unseen"]
    MB --> X
    X -- trains --> S["Association<br/>learned, Phase 3"]
    T -- trains --> F["Full model"]
    V["Validation half"] -. embedded by .-> F
    F -- test features --> S
```

This is the part I'd point at first.

**Why it's needed.** The association step is learned too. It looks at pairs of a known person and a new box, and learns from the training half how far to trust appearance similarity against position and timing. So it needs appearance vectors for the training half, and the obvious source is the fine-tuned model. That's the problem: the fine-tuned model trained on every person in the training half. I measured what its features look like on those same people:

| Training-half features made by | Fold A videos | Fold B videos | All seven, mAP | All seven, Rank-1 |
|---|---|---|---|---|
| OSNet x1.0 off the shelf, never saw MOT17 | 79.6 | 60.1 | 71.1 | 85.4 |
| Full fine-tuned model, trained on these people | **100.0** | **100.0** | **100.0** | **100.0** |
| Cross-fit, each video embedded by the other fold's model | 76.4 | 63.3 | 70.7 | 87.6 |

(mAP unless marked; same scoring rules as above, with every person in the training half as a query.)

**A perfect 100.** On the people it trained on, the full model practically never ranks a stranger above the right person. On new people it scores 77.9. An association step trained on those features would learn that appearance is never wrong, and then meet appearance that's wrong a fair share of the time. I found this the hard way: a matcher trained on them scored a perfect 100 average precision on its own held-out check. That's a leak, not a result.

The cross-fit features score **70.7** on the same people, a little below the 77.9 the full model reaches on new people. Each fold model trained on about half the data, so its features are slightly worse than the ones used at test time. That errs on the safe side: the association step learns to trust appearance a bit less than it could, rather than more. (The training-half crops are every frame and the validation ones every third frame, so the 70.7 and 77.9 are only roughly comparable.)

**What I did.**

1. Split the seven training videos into two folds by video. Each training video is a different scene, so no person lands in both folds, and each fold mixes static and moving cameras.
2. Trained one OSNet x1.0 per fold, with the full model's exact recipe and starting weights, on that fold's training-half crops only. They took 39 and 30 minutes, 69 in total, exactly as long as the full model took on its own.
3. Had each fold model embed every detection in the *other* fold's videos, and saved both halves into one shared set of features covering all seven videos.
4. The association step trains on those features. When tracking the validation half, it gets the full model's features instead.

The fold models have that one job. They aren't candidates for the tracker, and nothing is merged, averaged, or distilled from them. OSNet x0.5 and ResNet18 in the table above were trained separately, the same way as the full model.

**How the fold models perform.** On unseen validation people:

| Model | Trained on | Embeds | mAP | Rank-1 |
|---|---|---|---|---|
| Fold A | MOT17-04, 05, 11 | MOT17-02, 09, 10, 13 | 73.1 | 84.6 |
| Fold B | MOT17-02, 09, 10, 13 | MOT17-04, 05, 11 | 79.5 | 90.3 |
| Full | all seven | the validation half, at test time | 77.9 | 88.1 |

Each model is scored on the people *it* never saw, which is a different set for each, so these three rows can't be compared with each other. They only show that each model learned something. The training-half table is the fairer comparison, and there neither fold model is consistently better than the off-the-shelf one: model B scores 3.2 mAP below it on fold A's videos, and model A scores 3.2 above it on fold B's. That suggests three or four scenes of training don't carry far into scenes a model has never seen.

**What it changed.** With cross-fit features, the association step's held-out check reads **98.0** average precision (93.2% precision, 96.8% recall) instead of a suspicious 100.

The folds so far are OSNet x1.0's. With OSNet x0.5 chosen above, it needs a pair of its own.

## Usage

Requires [uv](https://docs.astral.sh/uv/). On Windows and Linux, `uv sync` installs PyTorch built for CUDA 13.0.

```bash
uv sync
```

### Data

Extract the MOT17 release so that `MOT17/train` and `MOT17/test` sit at the repository root, then:

```bash
uv run python -m reidtrack.data.prepare --raw MOT17 --out data/mot17
uv run python -m reidtrack.data.stats --root data/mot17
```

`prepare` verifies and deduplicates the release into `data/mot17` and exports the splits for evaluation. It writes nothing if the copies differ. Afterwards `MOT17/` can be deleted. `stats` prints the per-sequence figures quoted above. MOT17 is not part of this repository and remains under its own terms.

### Baselines

```bash
uv run python -m reidtrack.retrieval.cache --weights data/weights/osnet_x1_0_msmt17.pth
uv run python -m reidtrack.baselines sort --min-score 0.5
uv run python -m reidtrack.baselines bytetrack
uv run python -m reidtrack.baselines deepsort --min-score 0.5 --max-cosine 0.2
```

`cache` embeds every public detection with OSNet[^osnet] and stores the vectors under `data/mot17/cache/`. The MSMT17-trained weights come from the torchreid model zoo[^torchreid]. `baselines` runs my reimplementations of SORT[^sort], ByteTrack[^bytetrack] and DeepSORT[^deepsort]. Results and scores go to `runs/<name>/`.

### Appearance model

```bash
uv run python -m reidtrack.retrieval.crops --split train_half
uv run python -m reidtrack.retrieval.crops --split val_half --stride 3
uv run python -m reidtrack.retrieval.bench
uv run python -m reidtrack.retrieval.train --backbone osnet_x1_0 --init data/weights/osnet_x1_0_msmt17.pth
uv run python -m reidtrack.retrieval.cache --weights data/weights/retriever/osnet_x1_0_mot17/last.pt --model osnet_x1_0_mot17
```

- **`crops`** cuts the training and scoring crops.
- **`bench`** times candidate models.
- **`train`** fine-tunes one. Press Ctrl+C, or create a file named `PAUSE` in its run folder, to pause; run the same command with `--resume` to continue.
- **`cache`** embeds every detection with the fine-tuned model.

Cross-fitting trains one model per fold and has each embed the other fold into a shared cache:

```bash
uv run python -m reidtrack.retrieval.train --init data/weights/osnet_x1_0_msmt17.pth --name osnet_x1_0_foldA --sequences MOT17-04,MOT17-05,MOT17-11
uv run python -m reidtrack.retrieval.train --init data/weights/osnet_x1_0_msmt17.pth --name osnet_x1_0_foldB --sequences MOT17-02,MOT17-09,MOT17-10,MOT17-13
uv run python -m reidtrack.retrieval.cache --weights data/weights/retriever/osnet_x1_0_foldA/last.pt --model osnet_x1_0_crossfit --sequences MOT17-02,MOT17-09,MOT17-10,MOT17-13
uv run python -m reidtrack.retrieval.cache --weights data/weights/retriever/osnet_x1_0_foldB/last.pt --model osnet_x1_0_crossfit --sequences MOT17-04,MOT17-05,MOT17-11
```

### Evaluation

```bash
uv run python -m reidtrack.eval --results runs/<tracker> --split val_half
uv run python -m reidtrack.eval --oracle
```

Result files are one `<sequence>.txt` per sequence in MOT format, using the sequence's own frame numbers; rows outside the split are ignored. Scoring uses TrackEval[^hota]. `--oracle` scores the ground truth itself and must report 100.

### Visualization

```bash
uv run python -m reidtrack.viz MOT17-02 --gt --split val_half --scale 0.5
uv run python -m reidtrack.viz MOT17-02 --results runs/<tracker>/MOT17-02.txt --frame 340
```

Writes an MP4, or a PNG for a single `--frame`, to `runs/viz/`. People who are annotated but hidden are drawn dashed.

### Latency

```bash
uv run python -m reidtrack.eval.latency --seq MOT17-04
```

Benchmarks reading and decoding frames. `StageTimer` in `reidtrack.eval.latency` times the stages of a pipeline.

### Tests

```bash
uv run pytest
```

## License

Apache-2.0; see [LICENSE](LICENSE). `src/reidtrack/retrieval/osnet.py` is adapted from deep-person-reid[^torchreid] under the MIT License, whose notice is kept in that file. Pretrained weights and datasets are not included and remain under their own terms.

[^kuhn]: H. W. Kuhn. The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1–2):83–97, 1955.
[^stats]: Measured on the MOT17 training set with `python -m reidtrack.data.stats`. An occlusion here is a stretch in which a person's visibility drops below 0.1 and later recovers.
[^bytetrack]: Y. Zhang et al. ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *ECCV*, 2022. [arXiv:2110.06864](https://arxiv.org/abs/2110.06864). The reference implementation keeps a lost track for 30 frames (`track_buffer`), scaled to the frame rate.
[^mot16]: A. Milan, L. Leal-Taixé, I. Reid, S. Roth, K. Schindler. MOT16: A Benchmark for Multi-Object Tracking. [arXiv:1603.00831](https://arxiv.org/abs/1603.00831), 2016. Annotation rules in §2; evaluation classes in §4.
[^fasterrcnn]: S. Ren, K. He, R. Girshick, J. Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS*, 2015. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497).
[^dpm]: P. F. Felzenszwalb, R. B. Girshick, D. McAllester, D. Ramanan. Object Detection with Discriminatively Trained Part-Based Models. *IEEE TPAMI*, 32(9):1627–1645, 2010.
[^sdp]: F. Yang, W. Choi, Y. Lin. Exploit All the Layers: Fast and Accurate CNN Object Detector with Scale Dependent Pooling and Cascaded Rejection Classifiers. *CVPR*, 2016.
[^centertrack]: X. Zhou, V. Koltun, P. Krähenbühl. Tracking Objects as Points. *ECCV*, 2020. [arXiv:2004.01177](https://arxiv.org/abs/2004.01177).
[^hota]: J. Luiten et al. HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking. *IJCV*, 2021. [arXiv:2009.07736](https://arxiv.org/abs/2009.07736). Computed with [TrackEval](https://github.com/JonathonLuiten/TrackEval).
[^clear]: K. Bernardin, R. Stiefelhagen. Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics. *EURASIP Journal on Image and Video Processing*, 2008.
[^idf1]: E. Ristani, F. Solera, R. Zou, R. Cucchiara, C. Tomasi. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. *ECCV Workshops*, 2016. [arXiv:1609.01775](https://arxiv.org/abs/1609.01775).
[^sort]: A. Bewley, Z. Ge, L. Ott, F. Ramos, B. Upcroft. Simple Online and Realtime Tracking. *ICIP*, 2016. [arXiv:1602.00763](https://arxiv.org/abs/1602.00763).
[^deepsort]: N. Wojke, A. Bewley, D. Paulus. Simple Online and Realtime Tracking with a Deep Association Metric. *ICIP*, 2017. [arXiv:1703.07402](https://arxiv.org/abs/1703.07402).
[^osnet]: K. Zhou, Y. Yang, A. Cavallaro, T. Xiang. Omni-Scale Feature Learning for Person Re-Identification. *ICCV*, 2019. [arXiv:1905.00953](https://arxiv.org/abs/1905.00953).
[^torchreid]: K. Zhou, T. Xiang. Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch. [arXiv:1910.10093](https://arxiv.org/abs/1910.10093), 2019. Code and model zoo: [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid).
[^msmt17]: L. Wei, S. Zhang, W. Gao, Q. Tian. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. *CVPR*, 2018. [arXiv:1711.08565](https://arxiv.org/abs/1711.08565).
[^resnet]: K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. *CVPR*, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
[^bagoftricks]: H. Luo, Y. Gu, X. Liao, S. Lai, W. Jiang. Bag of Tricks and A Strong Baseline for Deep Person Re-identification. *CVPR Workshops*, 2019. [arXiv:1903.07071](https://arxiv.org/abs/1903.07071).
[^triplet]: A. Hermans, L. Beyer, B. Leibe. In Defense of the Triplet Loss for Person Re-Identification. [arXiv:1703.07737](https://arxiv.org/abs/1703.07737), 2017.
[^erasing]: Z. Zhong, L. Zheng, G. Kang, S. Li, Y. Yang. Random Erasing Data Augmentation. *AAAI*, 2020. [arXiv:1708.04896](https://arxiv.org/abs/1708.04896).
