import torch
import numpy as np
import fairseq.tasks.translation
import fairseq.data.data_utils


def collate(values, pad_idx, eos_idx):
    size = max(v.size(0) for v in values)
    size += 1
    res = values[0].new(len(values), size).fill_(pad_idx)
    for i, v in enumerate(values):
        res[i][0] = eos_idx
        res[i][1:len(v)+1].copy_(v)
    return res


class ConquestDataset(fairseq.data.FairseqDataset):
    def __init__(self, src, tgt, pad_idx, eos_idx, shuffle=True):
        assert len(src) == len(tgt)
        self.src = src
        self.tgt = tgt
        indices = []
        src_sizes = []
        tgt_sizes = []
        for i, (s, t) in enumerate(zip(src, tgt)):
            size = t.size(0)
            indices.extend([i] * size)
            tgt_sizes.extend(list(range(1, size + 1)))
            src_sizes.extend([s.size(0)] * size)
        self.shuffle = shuffle
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

        self.indices = np.array(indices)
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes)

    def __getitem__(self, idx):
        index = self.indices[idx]
        tgt_size = self.tgt_sizes[idx]
        src_tensor = self.src[index]
        tgt_tensor = self.tgt[index]
        return {
            'id': idx,
            'source': src_tensor,
            'target': tgt_tensor[:tgt_size]
        }

    def __len__(self):
        return len(self.indices)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        src_tokens = [x['source'] for x in samples]
        tgt_tokens = [x['target'] for x in samples]
        src_tokens = fairseq.data.data_utils.collate_tokens(src_tokens,
            pad_idx=self.pad_idx, eos_idx=self.eos_idx)
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        tgt_tokens = collate(tgt_tokens, pad_idx=self.pad_idx,
            eos_idx=self.eos_idx)
        ntokens = sum(len(s['target']) for s in samples)

        return {
            'id': [x['id'] for x in samples],
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'prev_output_tokens': tgt_tokens,
            },
            'target': tgt_tokens[:, 1:],
        }

    def num_tokens(self, idx):
        return max(self.src_sizes[idx], self.tgt_sizes[idx])

    def size(self, index):
        return self.src_sizes[index], self.tgt_sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]


@fairseq.tasks.register_task('conquest')
class ConquestTask(fairseq.tasks.translation.TranslationTask):
    def load_dataset(self, split, epoch=0, **kwargs):
        super().load_dataset(split, epoch)
        pair = self.datasets[split]
        src, tgt = pair.src, pair.tgt
        self.datasets[split] = ConquestDataset(src, tgt,
            pad_idx=pair.src_dict.pad(), eos_idx=pair.src_dict.eos())
