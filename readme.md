# Outlier Suppression
PyTorch implementation of Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models

## Overview
Outlier Suppression is an outlier suppression framework to overcome the quantization bottleneck of Transformer language models, including Gamma Migration and then Token-Wise Clipping. Gamma Migration utilizes migration equivalence to move the outlier amplifier to subsequent branches without any extra computation cost, avoiding the amplification of outliers and contributing to a more quantization-friendly distribution. Token-Wise Clipping takes the large variance of token range into consideration and clips the unimportant values with high efficiency in a token-wise coarse-to-fine pipeline.



## Usage

Go into the exp directory and you can see run.sh and config.yaml. run.sh represents a example for BERT 6-6-6. 

run.sh
```
#!/bin/bash
PYTHONPATH=../../../../:$PYTHONPATH \
python ../solver/run_glue_quant.py --config config.yaml
```

The quantization config in config.yaml:
```
quant:
    a_qconfig:
        quantizer: LSQPlusFakeQuantize
        observer: EMAPruneMinMaxObserver
        bit: 6
        symmetric: False
        ch_axis: -1 # perlayer -1
    w_qconfig:
        quantizer: FixedFakeQuantize
        observer: MinMaxObserver
        bit: 6
        symmetric: True
        ch_axis: 0
    calibrate: 256
    except_quantizer: null
    is_remove_padding: True
    delay_ln: True
```
'a_qconfig' and 'w_qconfig' means the quantization setting of activations and weights(embeddings) respectively. You can set the 'observer' as 'EMAPruneMinMaxObserver' to use the Token-Wise Clipping. And set 'delay_ln' as 'True' to perform Gamma Migration.


