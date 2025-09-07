import tqdm
import numpy as np
import torch
import os
import ast
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import random
import datetime

class RecDataset(Dataset):
    def __init__(self, args, user_seq, time_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.time_seq = []
        self.max_len = args.max_seq_length
        self.user_ids = []
        self.contrastive_learning = args.model_type.lower() in ['fearec', 'duorec']
        self.data_type = data_type

        if self.data_type=='train':
            for user, seq in enumerate(user_seq):
                """
                By default the original implementation excludes the last two
                interactions of each user (validation and test items) from the
                training data. To intentionally leak a portion of the test data
                into the training set we allow including the last interaction
                for a random subset of users controlled by
                ``args.test_train_ratio``.

                When ``test_train_ratio`` > 0 and the current user is selected,
                we keep the entire sequence (including the test item) for
                training. Otherwise we follow the original behaviour and remove
                the last two items.
                """

                if getattr(args, 'test_train_ratio', 0) > 0 \
                        and random.random() < args.test_train_ratio:
                    # Include the test item in the training sequence
                    input_ids = seq[-(self.max_len + 1):]
                    input_times = time_seq[user][-(self.max_len + 1):]
                else:
                    # Original behaviour: exclude validation and test items
                    input_ids = seq[-(self.max_len + 2):-2]
                    input_times = time_seq[user][-(self.max_len + 2):-2]

                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1])
                    self.time_seq.append(input_times[:i + 1])
                    self.user_ids.append(user)
        elif self.data_type=='valid':
            for sequence, t_seq in zip(user_seq, time_seq):
                self.user_seq.append(sequence[:-1])
                self.time_seq.append(t_seq[:-1])
        else:
            self.user_seq = user_seq
            self.time_seq = time_seq

        self.test_neg_items = test_neg_items

        # load precomputed popularity encodings
        self.pop_long_table = None
        self.pop_short_table = None
        if getattr(self.args, "use_popularity", False):
            pop_path = os.path.join(
                self.args.popularity_dir,
                f"{self.args.data_name}_pop_linear.txt",
            )
            if not os.path.exists(pop_path):
                raise FileNotFoundError(f"Popularity encoding file not found: {pop_path}")
            with open(pop_path) as f:
                long_vec, short_vec = None, None
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        continue
                    item = int(parts[0])
                    long_vec = ast.literal_eval(parts[1])
                    short_vec = ast.literal_eval(parts[2])
                    if self.pop_long_table is None:
                        self.pop_long_dim = len(long_vec)
                        self.pop_short_dim = len(short_vec)
                        self.pop_long_table = np.zeros(
                            (self.args.item_size, self.pop_long_dim), dtype=np.float32
                        )
                        self.pop_short_table = np.zeros(
                            (self.args.item_size, self.pop_short_dim), dtype=np.float32
                        )
                    self.pop_long_table[item] = np.array(long_vec, dtype=np.float32)
                    self.pop_short_table[item] = np.array(short_vec, dtype=np.float32)
            setattr(self.args, "pop_long_dim", self.pop_long_dim)
            setattr(self.args, "pop_short_dim", self.pop_short_dim)

        if self.contrastive_learning and self.data_type=='train':
            if os.path.exists(args.same_target_path):
                self.same_target_index = np.load(args.same_target_path, allow_pickle=True)
            else:
                print("Start making same_target_index for contrastive learning")
                self.same_target_index = self.get_same_target_index()
                self.same_target_index = np.array(self.same_target_index)
                np.save(args.same_target_path, self.same_target_index)

    def get_same_target_index(self):
        num_items = max([max(v) for v in self.user_seq]) + 2
        same_target_index = [[] for _ in range(num_items)]
        
        user_seq = self.user_seq[:]
        tmp_user_seq = []
        for i in tqdm.tqdm(range(1, num_items)):
            for j in range(len(user_seq)):
                if user_seq[j][-1] == i:
                    same_target_index[i].append(user_seq[j])
                else:
                    tmp_user_seq.append(user_seq[j])
            user_seq = tmp_user_seq
            tmp_user_seq = []

        return same_target_index

    def __len__(self):
        return len(self.user_seq)

    def _convert_time(self, timestamps):
        """Convert unix timestamps to month and week indices."""
        time1_seq, time2_seq = [], []
        for t in timestamps:
            if t > 0:
                dt = datetime.datetime.fromtimestamp(t)
                time1_seq.append(dt.year * 12 + dt.month)
                time2_seq.append(dt.isocalendar()[0] * 53 + dt.isocalendar()[1])
            else:
                time1_seq.append(0)
                time2_seq.append(0)
        return time1_seq, time2_seq

    def __getitem__(self, index):
        items = self.user_seq[index]
        times = self.time_seq[index]
        input_ids = items[:-1]
        input_times = times[:-1]
        answer = items[-1]

        seq_set = set(items)
        neg_answer = neg_sample(seq_set, self.args.item_size)

        time1_seq, time2_seq = self._convert_time(input_times)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        time1_seq = [0] * pad_len + time1_seq
        time2_seq = [0] * pad_len + time2_seq
        input_ids = input_ids[-self.max_len:]
        time1_seq = time1_seq[-self.max_len:]
        time2_seq = time2_seq[-self.max_len:]
        assert len(input_ids) == self.max_len
        assert len(time1_seq) == self.max_len
        assert len(time2_seq) == self.max_len

        if getattr(self.args, "use_popularity", False):
            pop_long_seq = [self.pop_long_table[i] for i in input_ids]
            pop_short_seq = [self.pop_short_table[i] for i in input_ids]

        if self.data_type in ['valid', 'test']:
            if getattr(self.args, "use_popularity", False):
                cur_tensors = (
                    torch.tensor(index, dtype=torch.long),  # user_id for testing
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(time1_seq, dtype=torch.long),
                    torch.tensor(time2_seq, dtype=torch.long),
                    torch.tensor(pop_long_seq, dtype=torch.float),
                    torch.tensor(pop_short_seq, dtype=torch.float),
                    torch.tensor(answer, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),  # not used
                    torch.zeros(0, dtype=torch.long),  # not used
                )
            else:
                cur_tensors = (
                    torch.tensor(index, dtype=torch.long),  # user_id for testing
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(time1_seq, dtype=torch.long),
                    torch.tensor(time2_seq, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),  # not used
                    torch.zeros(0, dtype=torch.long),  # not used
                )

        elif self.contrastive_learning:
            sem_augs = self.same_target_index[answer]
            sem_aug = random.choice(sem_augs)
            keep_random = False
            for i in range(len(sem_augs)):
                if sem_augs[0] != sem_augs[i]:
                    keep_random = True

            while keep_random and sem_aug == items:
                sem_aug = random.choice(sem_augs)

            sem_aug = sem_aug[:-1]
            pad_len = self.max_len - len(sem_aug)
            sem_aug = [0] * pad_len + sem_aug
            sem_aug = sem_aug[-self.max_len:]
            assert len(sem_aug) == self.max_len

            if getattr(self.args, "use_popularity", False):
                cur_tensors = (
                    torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(time1_seq, dtype=torch.long),
                    torch.tensor(time2_seq, dtype=torch.long),
                    torch.tensor(pop_long_seq, dtype=torch.float),
                    torch.tensor(pop_short_seq, dtype=torch.float),
                    torch.tensor(answer, dtype=torch.long),
                    torch.tensor(neg_answer, dtype=torch.long),
                    torch.tensor(sem_aug, dtype=torch.long)
                )
            else:
                cur_tensors = (
                    torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(time1_seq, dtype=torch.long),
                    torch.tensor(time2_seq, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                    torch.tensor(neg_answer, dtype=torch.long),
                    torch.tensor(sem_aug, dtype=torch.long)
                )

        else:
            if getattr(self.args, "use_popularity", False):
                cur_tensors = (
                    torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(time1_seq, dtype=torch.long),
                    torch.tensor(time2_seq, dtype=torch.long),
                    torch.tensor(pop_long_seq, dtype=torch.float),
                    torch.tensor(pop_short_seq, dtype=torch.float),
                    torch.tensor(answer, dtype=torch.long),
                    torch.tensor(neg_answer, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),  # not used
                )
            else:
                cur_tensors = (
                    torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(time1_seq, dtype=torch.long),
                    torch.tensor(time2_seq, dtype=torch.long),
                    torch.tensor(answer, dtype=torch.long),
                    torch.tensor(neg_answer, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),  # not used
                )

        return cur_tensors


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_rating_matrix(data_name, seq_dic, max_item):
    
    num_items = max_item + 1
    valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['num_users'], num_items)
    test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['num_users'], num_items)

    return valid_rating_matrix, test_rating_matrix

def get_user_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()
    lines = lines[1:]
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split('	', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, max_item

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    time_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        item_list = []
        time_list = []
        for it in items:
            if ':' in it:
                i, t = it.split(':')
            elif ',' in it:
                i, t = it.split(',')
            else:
                i, t = it, 0
            item_list.append(int(i))
            time_list.append(int(t))
            item_set.add(int(i))
        user_seq.append(item_list)
        time_seq.append(time_list)
    max_item = max(item_set)
    num_users = len(lines)

    return user_seq, time_seq, max_item, num_users

def get_seq_dic(args):

    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, time_seq, max_item, num_users = get_user_seqs(args.data_file)
    seq_dic = {'user_seq': user_seq, 'time_seq': time_seq, 'num_users': num_users}

    return seq_dic, max_item, num_users

def get_dataloder(args, seq_dic):

    train_dataset = RecDataset(args, seq_dic['user_seq'], seq_dic['time_seq'], data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    eval_dataset = RecDataset(args, seq_dic['user_seq'], seq_dic['time_seq'], data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    test_dataset = RecDataset(args, seq_dic['user_seq'], seq_dic['time_seq'], data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_dataloader, eval_dataloader, test_dataloader
