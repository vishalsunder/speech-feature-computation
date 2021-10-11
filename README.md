# Speech-feature-computation


Input is expected as a csv file with the following columns:

| cnum | utt_id | speaker | audio_file | utterance | begin | end |
| ---- | ------ | ------- | ---------- | --------- | ----- | --- |


`cnum` - conversation number

`utt_id` - utterance ID

`speaker` - speaker ('A' or 'B'; default to 'A')

`audio_file` - complete path to where the audio is saved

`utterance` - transcript (optional)

`begin` - start of the utterance in the audio in seconds (default to 0)

`end` - end of the utterance in the audio in seconds (default to 0)

`dump.py` will save the speech signal as a `npz` array in the specified location

`data.py` has the `SpeechDataset(torch.utils.data.Dataset)` class which computes logmels on the fly using the `.npz` files during training by using `torch.utils.data.DataLoader`.
