# Third-party notices

This repository keeps only the source files needed to reproduce the reported
experiments. The original research repositories are not required at runtime.

## RobustX

The model-change and counterfactual code in `tae_workshop/methods/` was adapted
from <https://github.com/RobustCounterfactualX/RobustX>, commit
`f503bfacfc40e930fa6a5a9219eb1633eea25001`. The retained code implements
KD-tree counterfactuals, Wachter, RNCE, MCE, and their shared interval/MILP
machinery.

MIT License

Copyright (c) 2024 Aaryan Purohit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Robust Bayesian Recourse (RBR)

The local implementation in `tae_workshop/methods/rbr_core.py` and
`tae_workshop/methods/rbr_loss.py` is adapted from
<https://github.com/VinAIResearch/robust-bayesian-recourse>, commit
`8eaa8c27a5f6c61f226267b5dd124250e9c6b419`.

Copyright (c) 2022, VinAI. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

## Datasets with categorical features

The following files in `data/` were downloaded from the UCI Machine Learning
Repository on 2026-10-11 and are redistributed under the Creative Commons
Attribution 4.0 International licence (CC BY 4.0,
<https://creativecommons.org/licenses/by/4.0/>). They were converted to CSV
without changing any value; all cleaning, recoding and subsampling happens in
`robustness_benchmark/core/data.py`.

### South German Credit (`data/german_credit.csv`)

South German Credit [Dataset] (2019). UCI Machine Learning Repository.
<https://doi.org/10.24432/C5X89F>. Documented in Grömping, U. (2019), South
German Credit Data: Correcting a Widely Used Data Set, Report 4/2019, Beuth
University of Applied Sciences Berlin. Source archive:
<https://archive.ics.uci.edu/static/public/522/south+german+credit.zip>
(`SouthGermanCredit.asc`, SHA-256
`5f363343f356ca38a0236baab849e472846399b2176ccc5bd686483dd8a7562f`). Written as
CSV with the English column names of the archive's `read_SouthGermanCredit.R`.

SHA-256 `a4e28e9ca9e250ca619b052f73bf63c1f4ab9c8e883a2b944f68251ed55a02f3`
### Adult (`data/adult.csv`)

Becker, B. and Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning
Repository. <https://doi.org/10.24432/C5XW20>. Source archive:
<https://archive.ics.uci.edu/static/public/2/adult.zip> (`adult.data`, SHA-256
`5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d`;
`adult.test`, SHA-256
`a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05`). The two
files are concatenated (48,842 rows; the comment line of `adult.test` skipped)
with the column names of `adult.names` and leading spaces stripped.

SHA-256 `f7acee69ac6292d98dd6d2a3a87bf2f5496c850ca5414c565be3adcbb702d10b`

### Default of Credit Card Clients (`data/taiwan_default.csv`)

Yeh, I.-C. (2009). Default of Credit Card Clients [Dataset]. UCI Machine
Learning Repository. <https://doi.org/10.24432/C55S3H>. Source archive:
<https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip>
(`default of credit card clients.xls`, SHA-256
`30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`). The sheet
is written as CSV with its second header row (`ID`, `LIMIT_BAL`, ...,
`default payment next month`) as column names.

SHA-256 `4edce6da569b2c30788a85fd98a9bafb1a983e8ca1a7aac875dd7ff853cbd0bb`
