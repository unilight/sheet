# MOS-Bench

Last updated: September 2025

We provide a (possibly) easier-to-read [Google Spreadsheet file](https://docs.google.com/spreadsheets/d/1Uqi6upfJHasoduuY72_75qphJgU-PuY-u3OKBBMNaKI/edit?usp=sharing).

## Training sets

| Name                         | Type                                                   | Language             | FS (kHz) | \# samples (train/dev) |
| ---------------------------- | ------------------------------------------------------ | -------------------- | -------- | ---------------------- |
| BVCC                         | TTS, VC, natural speech                                | English              | 16       | 4944/1066              |
| SOMOS                        | TTS, natural speech                                    | English              | 24       | 14100/3000             |
| SingMOS                      | SVS, SVC, natural singing voice                        | Chinese, Japanese    | 16       | 2000/544               |
| NISQA                        | artificial & real distorted speech, clean speech       | English              | 48       | 11020/2700             |
| TMHINT-QI                    | artificial noisy speech, enhanced speech, clean speech | Chinese              | 16       | 11644/1293             |
| Tencent                      | artificial distorted speech, clean speech              | Chinese              | 16       | 10408/1155             |
| PSTN                         | PSTN speech, artificial distorted speech               | English              | 8        | 52839/5870             |
| URGENT2024-MOS               | artificial & real distorted speech, enhanced speech    | English              | 8-48     | 6210/690               |

### Details (WIP)

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


## Test sets

| Name                         | Type                                                   | Language             | FS (kHz) | \# samples (train/dev) |
| ---------------------------- | ------------------------------------------------------ | -------------------- | -------- | ---------------------- |
| BVCC test (VMC22 main track) | TTS, VC, natural speech                                | English              | 16       | 1066                   |
| SOMOS test                   | TTS, natural speech                                    | English              | 24       | 3000                   |
| BC19 (VMC22 OOD track)       | TTS, natural speech                                    | Chinese              | 16       | 540                    |
| BC23 Hub (VMC23 track1a)     | TTS, natural speech                                    | France               | 22       | 882                    |
| BC23 Spoke (VMC23 track1b)   | TTS, natural speech                                    | France               | 22       | 578                    |
| SVCC23 (VMC23 track2)        | SVC, natural singing voice                             | English              | 24       | 4040                   |
| SingMOS test (VMC24 track2)  | SVS, SVC, natural singing voice                        | Chinese, Japanese    | 16       | 645                    |
| BRSpeechMOS                  | TTS, natural speech                                    | Brazilian-Portuguese | 16       | 243                    |
| HablaMOS                     | TTS, natural speech                                    | Spanish              | 16       | 408                    |
| TTSDS2                       | TTS                                                    | English              | 22       | 4731                   |
| NISQA TEST FOR               | artificial distorted speech, VoIP                      | English              | 48       | 240                    |
| NISQA TEST LIVETALK          | real-world distorted speech, VoIP                      | Dutch                | 48       | 232                    |
| NISQA TEST P501              | artificial distorted speech, VoIP                      | English              | 48       | 240                    |
| TMHINT-QI test               | artificial noisy speech, enhanced speech, clean speech | Chinese              | 16       | 1978                   |
| TMHINT-QI(S) (VMC23 track3)  | artificial noisy speech, enhanced speech, clean speech | Chinese              | 16       | 1960                   |
| TCD-VOIP                     | artificial distorted speech, VoIP                      | English              | 48       | 384                    |
| VMC24 track3                 | artificial noisy speech, enhanced speech, clean speech | English              | 16       | 280                    |

### Details (WIP)

- BVCC test
    - See descriptions in the [Training datasets in MOS-Bench](#training-datasets-in-mos-bench) section.
    - Benchmark script: `BENCHMARKS/run_bvcc_test.sh`
- BC19 test (VMC'22 OOD track)
    - Dataset download link: https://zenodo.org/records/6572573
    - Paper link: [[VoiceMOS Challenge 2022](https://arxiv.org/abs/2203.11389)]
    - Benchmark script: `BENCHMARKS/run_bc19_test.sh`
- SOMOS test
    - See descriptions in the [Training datasets in MOS-Bench](#training-datasets-in-mos-bench) section.
    - Benchmark script: `BENCHMARKS/run_somos_test.sh`
- SingMOS test (VMC'24 track 2)
    - See descriptions in the [Training datasets in MOS-Bench](#training-datasets-in-mos-bench) section.
    - Benchmark script: `BENCHMARKS/run_singmos_test.sh`
- NISQA FOR/P501/LIVETALK
    - See descriptions in the [Training datasets in MOS-Bench](#training-datasets-in-mos-bench) section.
    - Benchmark script: `BENCHMARKS/run_nisqa_test.sh`
- TMHINT-QI test
    - See descriptions in the [Training datasets in MOS-Bench](#training-datasets-in-mos-bench) section.
    - Benchmark script: `BENCHMARKS/run_tmhint_qi_test.sh`
- VMC'23 track 1a/1b/2/3 (BC2023, SVCC2023, TMHINTQI-(S))
    - Paper link: [[arXiv version](https://arxiv.org/abs/2310.02640)]
    - Benchmark script: `BENCHMARK/run_vmc23_test.sh`