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