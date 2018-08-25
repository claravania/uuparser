# UUParser-CPH

This is a fork of [UUParser](https://github.com/UppsalaNLP/uuparser) modified for the paper:

Parameter sharing between dependency parsers for related languages, by Miryam de Lhoneux, Johannes Bjerva, Isabelle Augenstein and Anders Søgaard at EMNLP 2018.

Installation and usage instructions can be found on the original github repository. Our experimental settings can be used by adding options to parser training/prediction. There are 3 options `--word`, `--char` and `--mlp` which can be set to `shared`, `not_shared` or `shared_lembed` which correspond to hard sharing, not sharing and sharing with a language embedding of words, characters and the MLP respectively.
Note that the parser needs to be run with the option `--multiling` to train in multilingual mode.

To find out about more options, run

```
python barchybrid/src/parser.py --help
```


#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

#### Contact

For questions and usage issues, please contact miryam dot de underscore lhoneux at lingfil dot uu dot se
