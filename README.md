# Speech-feature-computation

* Please download sph2pipe from [URL](https://www.openslr.org/3/)
* Installation guide [URL](https://github.com/robd003/sph2pipe)

`dump.py` will save the speech signal as a `npz` array in the specified location

`data.py` has the `SpeechDataset(torch.utils.data.Dataset)` class which computes logmels on the fly during training using the `.npz` files during training by using `torch.utils.data.DataLoader`. Feel free to modify for your training pipeline.

Change the file paths in `dump.py` and `data.py`.

Input (`path_csv` in `dump.py`) is expected as a csv file with the following columns:

| cnum | utt_id | speaker | audio_file | utterance | begin | end |
| ---- | ------ | ------- | ---------- | --------- | ----- | --- |


`cnum` - conversation number

`utt_id` - utterance ID

`speaker` - speaker ('A' or 'B'; default to 'A')

`audio_file` - complete path to where the audio is saved

`utterance` - transcript (optional)

`begin` - start of the utterance in the audio in seconds (default to 0)

`end` - end of the utterance in the audio in seconds (default to 0)
