## Supported Training datasets

- BVCC
    - Dataset download link: https://zenodo.org/records/6572573
    - Paper link: [[Original paper](https://arxiv.org/abs/2105.02373)] [[VoiceMOS Challenge 2022](https://arxiv.org/abs/2203.11389)]
    - Recipe: `egs/bvcc`
- SOMOS
    - Dataset download link: https://zenodo.org/records/7378801
    - Paper link: [[arXiv version](https://arxiv.org/abs/2204.03040)]
    - Recipe: `egs/somos`
- NISQA
    - Dataset download link: https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus
    - Paper link: [[arXiv version](https://arxiv.org/abs/2104.09494)]
    - Recipe: `egs/nisqa`
- TMHINT-QI
    - Dataset download link: https://drive.google.com/file/d/1TMDiz6dnS76hxyeAcCQxeSqqEOH4UDN0/view?usp=sharing
    - Paper link: [[INTERSPEECH 2022 version](https://www.isca-speech.org/archive/pdfs/interspeech_2022/chen22i_interspeech.pdf)]
    - Recipe: `egs/tmhint-qi`
- PSTN
    - Dataset download link: https://challenge.blob.core.windows.net/pstn/train.zip
    - Paper link: [[arXiv version](https://arxiv.org/abs/2007.14598)]
    - Recipe: `egs/pstn`
- Tencent
    - Dataset download link: https://www.dropbox.com/s/ocmn78uh2lu5iwg/TencentCorups.zip?dl=0
    - Paper link: [[arXiv version](https://arxiv.org/abs/2203.16032)]
    - Recipe: `egs/tencent`
- SingMOS
    - Dataset download link: https://drive.google.com/file/d/1DtzZhk3M_jsxUxirPcFRoBhq-dsinOWN/view?usp=drive_link
    - Paper link: [[arXiv version](https://arxiv.org/abs/2406.10911)]
    - Recipe: `egs/singmos`

## Supported Benchmarks

In addition to the test sets provided in the datasets above, you can do zero-shot evaluation on the following benchmarks.
For usage, see [egs/BENCHMARK/README.md](egs/BENCHMARK)

- VoiceMOS Challenge 2023 (BC2023, SVCC2023, TMHINTQI-(S))
    - Paper link: [[arXiv version](https://arxiv.org/abs/2310.02640)]
    - Recipe: `egs/BENCHMARK/run_vmc23_test.sh`

Of course you can also do zero-shot benchmarking on the test sets of the training datasets:

- BVCC: `egs/BENCHMARK/run_bvcc_test.sh`
- SOMOS: `egs/BENCHMARK/run_somos_test.sh`
- NISQA: `egs/BENCHMARK/run_nisqa_test.sh`
    
In the future, we plan to support the following additional benchmarks:

- VoiceMOS Challenge 2022 OOD track
- VoiceMOS Challenge 2024

## Basic Knowledge

### Data format

The common data format across all recipes is csv. Each csv file MUST have the following columns:

- `wav_path`
- `score`
- `system_id`
- `sample_id`

Optionally, the following columns may exist:

- `listener_id`
- `listener_idx`
- `phoneme`
- `cluster`
- `reference`

### Training a speech quality predictor with a supported dataset on your own

You can train your own speech quality predictor using the datasets we support. Usually these datasets come with their own test sets, and you can test on these sets after training finishes. The starting point of each recipe is the `run.sh` file. Please check the detailed usage in each recipe.

### Zero-shot prediction on multiple benchmarks with your trained model

Inside each recipe, after the model training is done, you can run zero-shot prediction on other datasets and benchmarks. This can be done by running `run_XXX_test.sh` in each recipe. They are symbolic links to the scripts in the `egs/BENCHMARKS` folder.


## Supported datasets and benchmarks

| Name                          | Recipe | Domain                                    | Language           | # Samples |
|-------------------------------|--------|-------------------------------------------|--------------------|-----------|
| BVCC train                    |        | TTS,VC,Clean speech                       | English            | 4944      |
| SOMOS train                   |        | TTS,Clean speech                          | English            | 14100     |
| NISQA train                   |        | Noisy speech                              | English            | 10000     |
| TMHINT-QI train               |        | Noisy speech,Enhanced speech,Clean speech | Taiwanese Mandarin | 11644     |
| SingMOS-VMC24 train           |        | Clean singing voice,SVS,SVC               | Chinese,Japanese   | 2000      |
| Tencent train                 |        | Noisy speech                              | Chinese            |           |
| PSTN train                    |        | VoIP calls,Noisy speech                   | English            | 58710     |
| BVCC test                     |        | TTS,VC,Clean speech                       | English            | 1066      |
| SOMOS test                    |        | TTS,Clean speech                          | English            | 3000      |
| NISQA TEST FOR                |        | Noisy speech                              | English            | 240       |
| NISQA TEST P501               |        | Noisy speech                              | English            | 240       |
| NISQA TEST LIVETALK           |        | Live noisy speech                         | English            | 232       |
| TMHINT-QI test                |        | Noisy speech,Enhanced speech,Clean speech | Taiwanese Mandarin | 1978      |
| VMC’23 track 2 (SingMOS test) |        | Clean singing voice,SVC,SVS               | Chinese,Japanese   | 645       |
| VMC’23 track 1 (BC’23)        |        | TTS,Clean speech                          | English            | 1460      |
| VMC’23 track 2 (SVCC’23)      |        | SVC,Clean singing voice                   | English            | 4040      |
| VMC’23 track 3 (TMHINT-QI 2)  |        | Noisy speech,Enhanced speech,Clean speech | Taiwanese Mandarin | 1960      |
|                               |        |                                           |                    |           |