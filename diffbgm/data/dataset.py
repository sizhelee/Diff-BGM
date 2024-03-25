# pyright: reportOptionalSubscript=false

import os
import torch
import numpy as np
import sys

# from transformers import BertTokenizer, BertModel

from torch.utils.data import Dataset
from utils import (
    nmat_to_pianotree_repr, prmat2c_to_midi_file, nmat_to_prmat2c, chd_to_midi_file,
    estx_to_midi_file, nmat_to_prmat, prmat_to_midi_file, show_image, chd_to_onehot
)
from utils import read_dict
from dirs import *

import math

SEG_LGTH = 32
N_BIN = 4
SEG_LGTH_BIN = SEG_LGTH * N_BIN

# tokenizer = BertTokenizer.from_pretrained('/network_space/storage43/lisizhe/models/bert-base-uncased')
# bert_model = BertModel.from_pretrained("/network_space/storage43/lisizhe/models/bert-base-uncased")


class DataSampleNpz:
    """
    A pair of song segment stored in .npz format
    containing piano and orchestration versions

    This class aims to get input samples for a single song
    `__getitem__` is used for retrieving ready-made input segments to the model
    it will be called in DataLoader
    """
    def __init__(self, song_fn, use_track=[0, 1, 2]) -> None:  # NOTE: use melody now!
        self.fpath = os.path.join(POP909_DATA_DIR, song_fn)
        self.song_fn = song_fn
        self.song_idx = song_fn[:song_fn.find(".")]
        """
        notes (onset_beat, onset_bin, duration, pitch, velocity)
        start_table : i-th row indicates the starting row of the "notes" array
            at i-th beat.
        db_pos: an array of downbeat beat_ids

        x: orchestra
        y: piano

        dict : each downbeat corresponds to a SEG_LGTH-long segment
            nmat: note matrix (same format as input npz files)
            pr_mat: piano roll matrix (the format for texture decoder)
            pnotree: pnotree format (used for calculating loss & teacher-forcing)
        """

        # self.notes = None
        # self.chord = None
        # self.start_table = None
        # self.db_pos = None

        # self._nmat_dict = None
        # self._pnotree_dict = None
        # self._pr_mat_dict = None
        # self._feat_dict = None

        # def load(self, use_chord=False):
        #     """ load data """
        self.use_track = use_track  # which tracks to use when converting to prmat2c

        data = np.load(self.fpath, allow_pickle=True)
        self.notes = np.array(
            data["notes"]
        )  # NOTE: here we have 3 tracks: melody, bridge and piano
        self.start_table = data["start_table"]  # NOTE: same here

        self.db_pos = data["db_pos"]
        self.db_pos_filter = data["db_pos_filter"]
        self.db_pos = self.db_pos[self.db_pos_filter]
        if len(self.db_pos) != 0:
            self.last_db = self.db_pos[-1]
        else:
            print(self.song_idx, self.db_pos)

        self.chord = data["chord"].astype(np.int32)
        self.visual = torch.load(f'./data/bgm909/new_raw_video_feats/{self.song_idx}.pth', map_location='cpu')
        self.caption = np.load(f'./data/bgm909/caption_feats/{self.song_idx}.npy')
        
        shot_path = f"./data/bgm909/detection/{self.song_idx}_scenes.txt"
        shot_cnt = 0
        with open(shot_path, "r") as f:
            while f.readline():
                shot_cnt += 1
        self.seg_shot_cnt = int(round(shot_cnt / len(self)))

        tmp = self.visual.shape[0] / self.chord.shape[0]
        idx_ls = []
        for i in range(self.chord.shape[0]):
            idx_ls.append(int(i*tmp))
        self.visual = torch.tensor(self.visual[idx_ls, :])

        tmp = self.caption.shape[0] / self.chord.shape[0]
        idx_ls = []
        for i in range(self.chord.shape[0]):
            idx_ls.append(int(i*tmp))
        self.caption = torch.tensor(self.caption[idx_ls, :])

        self._nmat_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._pnotree_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._prmat2c_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._prmat_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))

    def __len__(self):
        """Return number of complete 8-beat segments in a song"""
        return len(self.db_pos)

    def note_mat_seg_at_db(self, db):
        """
        Select rows (notes) of the note_mat which lie between beats
        [db: db + 8].
        """

        seg_mats = []
        for track_idx in self.use_track:
            notes = self.notes[track_idx]
            start_table = self.start_table[track_idx]

            s_ind = start_table[db]
            if db + SEG_LGTH_BIN in start_table:
                e_ind = start_table[db + SEG_LGTH_BIN]
                note_seg = np.array(notes[s_ind : e_ind])
            else:
                note_seg = np.array(notes[s_ind :])  # NOTE: may be wrong
            seg_mats.extend(note_seg)

        seg_mats = np.array(seg_mats)
        if seg_mats.size == 0:
            seg_mats = np.zeros([0, 5])
        return seg_mats

    @staticmethod
    def cat_note_mats(note_mats):
        return np.concatenate(note_mats, 0)

    @staticmethod
    def reset_db_to_zeros(note_mat, db):
        note_mat[:, 0] -= db

    @staticmethod
    def format_reset_seg_mat(seg_mat):
        """
        The input seg_mat is (N, 5)
            onset, pitch, duration, velocity, program = note
        The output seg_mat is (N, 3). Columns for onset, pitch, duration.
        Onset ranges between range(0, 32).
        """

        output_mat = np.zeros((len(seg_mat), 3), dtype=np.int64)
        output_mat[:, 0] = seg_mat[:, 0]
        output_mat[:, 1] = seg_mat[:, 1]
        output_mat[:, 2] = seg_mat[:, 2]
        return output_mat

    def store_nmat_seg(self, db):
        """
        Get note matrix (SEG_LGTH) of orchestra(x) at db position
        """
        if self._nmat_dict[db] is not None:
            return

        nmat = self.note_mat_seg_at_db(db)
        self.reset_db_to_zeros(nmat, db)

        nmat = self.format_reset_seg_mat(nmat)
        self._nmat_dict[db] = nmat

    def store_prmat2c_seg(self, db):
        """
        Get piano roll format (SEG_LGTH) from note matrices at db position
        """
        if self._prmat2c_dict[db] is not None:
            return

        prmat2c = nmat_to_prmat2c(self._nmat_dict[db], SEG_LGTH_BIN)
        self._prmat2c_dict[db] = prmat2c

    def store_prmat_seg(self, db):
        """
        Get piano roll format (SEG_LGTH) from note matrices at db position
        """
        if self._prmat_dict[db] is not None:
            return

        prmat2c = nmat_to_prmat(self._nmat_dict[db], SEG_LGTH_BIN)
        self._prmat_dict[db] = prmat2c

    def store_pnotree_seg(self, db):
        """
        Get pnotree representation (SEG_LGTH) from nmat
        """
        if self._pnotree_dict[db] is not None:
            return

        self._pnotree_dict[db] = nmat_to_pianotree_repr(
            self._nmat_dict[db], n_step=SEG_LGTH_BIN
        )

    def _store_seg(self, db):
        self.store_nmat_seg(db)
        self.store_prmat2c_seg(db)
        self.store_prmat_seg(db)
        self.store_pnotree_seg(db)

    def _get_item_by_db(self, db):
        """
        Return segments of
            prmat, prmat_y
        """

        self._store_seg(db)

        seg_prmat2c = self._prmat2c_dict[db]
        seg_prmat = self._prmat_dict[db]
        seg_pnotree = self._pnotree_dict[db]
        chord = self.chord[db // N_BIN : db // N_BIN + SEG_LGTH]
        if chord.shape[0] < SEG_LGTH:
            chord = np.append(
                chord,
                np.zeros([SEG_LGTH - chord.shape[0], 14], dtype=np.int32),
                axis=0
            )

        visual = self.visual.numpy()[db // N_BIN : db // N_BIN + SEG_LGTH]
        visual = np.append(
            visual,
            np.zeros([SEG_LGTH - visual.shape[0], visual.shape[1]], dtype=np.int32),
            axis=0
            )

        caption = self.caption.numpy()[db // N_BIN : db // N_BIN + SEG_LGTH]
        caption = np.append(
            caption,
            np.zeros([SEG_LGTH - caption.shape[0], caption.shape[1]], dtype=np.int32),
            axis=0
            )
        return seg_prmat2c, seg_pnotree, chord, seg_prmat, visual, caption, self.seg_shot_cnt

    def __getitem__(self, idx):
        db = self.db_pos[idx]
        return self._get_item_by_db(db)

    def get_whole_song_data(self):
        """
        used when inference
        """
        prmat2c = []
        pnotree = []
        chord = []
        prmat = []
        visual = []
        caption = []
        idx = 0
        i = 0
        while i < len(self):
            seg_prmat2c, seg_pnotree, seg_chord, seg_prmat, seg_visual, seg_caption, _ = self[i]
            prmat2c.append(seg_prmat2c)
            pnotree.append(seg_pnotree)
            chord.append(chd_to_onehot(seg_chord))
            prmat.append(seg_prmat)
            visual.append(seg_visual)
            caption.append(seg_caption)

            idx += SEG_LGTH_BIN
            while i < len(self) and self.db_pos[i] < idx:
                i += 1
        prmat2c = torch.from_numpy(np.array(prmat2c, dtype=np.float32))
        pnotree = torch.from_numpy(np.array(pnotree, dtype=np.int64))
        chord = torch.from_numpy(np.array(chord, dtype=np.float32))
        prmat = torch.from_numpy(np.array(prmat, dtype=np.float32))
        visual = torch.from_numpy(np.array(visual, dtype=np.float32))
        caption = torch.from_numpy(np.array(caption, dtype=np.float32))

        return prmat2c, pnotree, chord, prmat, visual, caption, self.seg_shot_cnt


class PianoOrchDataset(Dataset):
    def __init__(self, data_samples, debug=False):
        super(PianoOrchDataset, self).__init__()

        # a list of DataSampleNpz
        self.data_samples = data_samples

        self.lgths = np.array([len(d) for d in self.data_samples], dtype=np.int64)
        self.lgth_cumsum = np.cumsum(self.lgths)
        self.debug = debug

    def __len__(self):
        return self.lgth_cumsum[-1]

    def __getitem__(self, index):
        # song_no is the smallest id that > dataset_item
        song_no = np.where(self.lgth_cumsum > index)[0][0]
        song_item = index - np.insert(self.lgth_cumsum, 0, 0)[song_no]

        song_data = self.data_samples[song_no]
        if self.debug:
            return *song_data[song_item], song_data.song_fn
        else:
            return song_data[song_item]

    @classmethod
    def load_with_song_paths(cls, song_paths, debug=False, **kwargs):
        # data_samples = [DataSampleNpz(song_path, **kwargs) for song_path in song_paths]
        data_samples = []
        for song_path in song_paths:
            if song_path in ["203.npz", "215.npz", "254.npz", "280.npz", "328.npz", "034.npz"]:
                continue
            data_samples.append(DataSampleNpz(song_path, **kwargs))

        return cls(data_samples, debug)

    @classmethod
    def load_train_and_valid_sets(cls, debug=False, **kwargs):
        split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "pop909_new.pickle"))
        print("load train valid set with:", kwargs)
        return cls.load_with_song_paths(split[0], debug,
                                        **kwargs), cls.load_with_song_paths(
                                            split[1], debug, **kwargs
                                        )

    @classmethod
    def load_valid_set(cls, debug=False, **kwargs):
        split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "symmv.pickle"))
        return cls.load_with_song_paths(split[1], debug, **kwargs)

    @classmethod
    def load_with_train_valid_paths(cls, tv_song_paths, debug=False, **kwargs):
        return cls.load_with_song_paths(tv_song_paths[0], debug,
                                        **kwargs), cls.load_with_song_paths(
                                            tv_song_paths[1], debug, **kwargs
                                        )


if __name__ == "__main__":
    test = "200.npz"
    song = DataSampleNpz(test)
    os.system(f"cp {POP909_DATA_DIR}/{test[:-4]}_flatten.mid exp/copy.mid")
    prmat2c, pnotree, chord, prmat, visual, caption, shot_cnt = song.get_whole_song_data()
    print(prmat2c.shape)
    print(pnotree.shape)
    print(chord.shape)
    print(prmat.shape)
    print(visual.shape)
    print(caption.shape)
    # show_image(prmat2c[: 1], "exp/img/prmat2c_1.png")
    # show_image(prmat2c[1 : 2], "exp/img/prmat2c_2.png")
    prmat2c = prmat2c.cpu().numpy()
    pnotree = pnotree.cpu().numpy()
    chord = chord.cpu().numpy()
    prmat = prmat.cpu().numpy()
    prmat2c_to_midi_file(prmat2c, "exp/prmat2c.mid")
    estx_to_midi_file(pnotree, "exp/pnotree.mid")
    chd_to_midi_file(chord, "exp/chord.mid")
    prmat_to_midi_file(prmat, "exp/prmat.mid")
