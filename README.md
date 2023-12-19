# tcd-timit

## Data

Data used in this project is TCD-TIMIT [1], an audio-visual speech recognition dataset. Only audio part was in fact used for model training. Training data contained wav files with spoken sentences at sample rate of 48 kHz, as targets the text transcriptions on different levels(phonemes/characters) were used. Division on training/test data suggested by the dataset creators was used, validation data was randomly sampled 5% of training. The model was trained on MFCC coefficients calculated from audio signal. There was a try to add 1st and 2nd MFCC derivatives, but it didn't lead to any improvement in terms of phoneme/character error rate.

## Model

3 layers of Bidirectional LSTM were used to optimise the CTC loss function [4] in both tasks. There was a try to modify default torch LSTM module by adding peephole connections [3], but they didn't help much.

## Phoneme results

The best trained model obtained 34.11%/40.22%/39.50% phoneme error rate on train/validation/test. For this task the beem search decoding with depth 5 was used. Loss change during training is demonstrated on plot below.

<img src="https://github.com/gekas145/tcd-timit/blob/main/images/learning.png" alt="drawing" width="500" height="400"/>

## Characters results

This task was substantially harder, as network was forced to learn spelling of words, some of which were only present in one sentence. Therefore it was hard to await any meaningful word/character error rate. Still even in such circumstances the trained model was able to achieve some interesting results. It was able to transcribe some words correctly and to spell the others predicting similarly sounding words in their place.

CTC decoding was carried out using token passing algorithm [2] which only allows words from dictionary. Also approach with silence threshold [4] was tested, but as silence moments in speech do not always correspond with transitions between words it didn't yield any meaningful results. Introduction of space character in CTC vocabulary also didn't do any better as network was then forced to understand that current word has ended to insert the space which makes the task substantially harder.

## Sources

[1] N. Harte and E. Gillen, "TCD-TIMIT: An Audio-Visual Corpus of Continuous Speech," in IEEE Transactions on Multimedia, vol. 17, no. 5, pp. 603-615, May 2015, [https://doi.org/10.1109/TMM.2015.2407694](https://doi.org/10.1109/TMM.2015.2407694)

[2] Graves A., "Supervised Sequence Labelling with Recurrent Neural Networks", [https://www.cs.toronto.edu/~graves/preprint.pdf](https://www.cs.toronto.edu/~graves/preprint.pdf)

[3] Gers F., Schraudolph N. and Schmidhuber J., "Learning Precise Timing with LSTM Recurrent Networks" , [https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf)

[4] Graves A., Fernandez S., Gomez F. and Schmidhuber J., "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks", [https://www.cs.toronto.edu/~graves/icml_2006.pdf](https://www.cs.toronto.edu/~graves/icml_2006.pdf)


