# spatiotemporal-traffic-model

A temtative reproduction of the spatiotemporal prediction model presented in: 

> Yu, H. et al. "Spatiotemporal Recurrent Convolutional Networks for Traffic Prediction in Transportation Networks." Sensors 17 (2017)

This was implemented in the context of [Älypysäkki](https://projects.tuni.fi/alypysakki/) project, and the model was trained with GPS data obtained from [Journeys API](https://data.tampere.fi/data/en_GB/dataset/journeys-api).


The losses measured for the first 10 epochs:

> LOSS VAL: 2.020426704286365
> LOSS TRAIN: 0.9694194785788568
> ------------------
> LOSS VAL: 1.8975817662139889
> LOSS TRAIN: 0.7844324744219193
> ------------------
> LOSS VAL: 1.838629871723242
> LOSS TRAIN: 0.7011500656590215
> ------------------
> LOSS VAL: 1.779264768003486
> LOSS TRAIN: 0.6770079022162463
> ------------------
> LOSS VAL: 1.6129144496517256
> LOSS TRAIN: 0.6728439453845567
> ------------------
> LOSS VAL: 1.6614121579332277
> LOSS TRAIN: 0.663026861837352
> ------------------
> LOSS VAL: 1.4334587816847488
> LOSS TRAIN: 0.6228367803669244
> ------------------
> LOSS VAL: 1.3374343122704886
> LOSS TRAIN: 0.6284602218165674
> ------------------
> LOSS VAL: 1.3096007623244077
> LOSS TRAIN: 0.6012938981139087
> ------------------
> LOSS VAL: 1.2755749042553362
> LOSS TRAIN: 0.5897912865966646
> ------------------

