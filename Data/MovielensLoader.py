import numpy as np


class MovielensLoader:
    def __init__(self, dataset, split=0.0, min_sample_len=8, label_mode="last", positive_rating=1):
        """
        The loader will load rating sequences. The element in the sequence is index of the movie,
        whose name can be found in 'movies.dat' file
        for example, there are two rating sequences:
            [1, 2, 3, 4, 5] (user1's rating sequence)
            [5, 4, 3, 2, 1] (user2's rating sequence)
        If min_sample_len = 3, label_mode = 'last', split = 0.5, then the train set can be:
            [1, 2, 3], 4, 1
            [2, 3, 4], 5, 1
        and the test set can be:
            [5, 4, 3], 2, 2
            [4, 3, 2], 1, 2
        If label_mode = 'all', the train set can be:
            [1, 2, 3], [2, 3, 4], 1
            [2, 3, 4], [3, 4, 5], 1
        and the test set can be:
            [5, 4, 3], [4, 3, 2], 2
            [4, 3, 2], [3, 2, 1], 2
        If split = 0.0, means use all users' last rating as test set, then the train set can be:
            [1, 2, 3], 4
            [5, 4, 3], 2
        and the test can be:
            [2, 3, 4], 5
            [4, 3, 2], 1
        :param dataset: a string indicates which dataset, should be one of ["m-100k", "m-1m"]
        :param split: if split = 0 (as default), use all users' last rating as test set
        :param min_sample_len: the minimum length of one sample rating sequence
        :param label_mode: 'last' for only last rating is label
        :param positive_rating: the ratings <= positive rating will be ignored
        """
        dataset_dict = {
            "m-100k": ("./Data/Datasets/MovieLens/Raw/ml-latest-small/ratings.csv", ","),
            'm-1m': ("./Data/Datasets/MovieLens/Raw/ml-1m/ratings.dat", ","),
        }
        assert dataset in dataset_dict, "Invalid dataset"
        self.path = dataset_dict[dataset][0]
        self.delimiter = dataset_dict[dataset][1]
        self.user_rating_seqs = []
        self.n_items = 0
        self.n_users = 0
        self.train_test_split = split
        self.seq_len = min_sample_len
        self.label_mode = label_mode
        self._generate_rating_history_seqs(positive_rating, min_len=min_sample_len, train_test_split=split)

    def _generate_rating_history_seqs(self, positive_rating=1, split_time=3600000, min_len=8, train_test_split=0.9):
        user_rating_seqs = []
        n_items = 0
        n_users = 0
        with open(self.path, "r") as ratings:
            line = ratings.readline()
            current_seq = []
            last_timestamp = None
            while line != "":
                user, item, rating, timestamp = line[:-1].split(self.delimiter)
                # Convert to integers
                user = int(user)
                if user > n_users:
                    n_users = user
                item = int(item)
                if item > n_items:
                    n_items = item
                rating = float(rating)
                timestamp = int(timestamp)

                if rating < positive_rating:
                    line = ratings.readline()
                    continue
                if len(current_seq) == 0:
                    current_seq.append(user)
                    current_seq.append(item)
                    last_timestamp = timestamp
                else:
                    if user == current_seq[0] and timestamp - last_timestamp < split_time:
                        current_seq.append(item)
                        last_timestamp = timestamp
                    else:
                        user_rating_seqs.append(current_seq)
                        current_seq = [user, item]
                line = ratings.readline()
            user_rating_seqs.append(current_seq)
        if train_test_split == 0:
            rating_seq_minlen = min_len + 3
        else:
            rating_seq_minlen = min_len + 2
        user_rating_seqs = [seq for seq in user_rating_seqs if len(seq) >= rating_seq_minlen]
        np.random.shuffle(user_rating_seqs)
        print("Generated {0} sequences, total {1} items and {2} users".format(len(user_rating_seqs), n_items, n_users))
        print("Sequence length is at least {}".format(rating_seq_minlen - 1))
        self.user_rating_seqs = user_rating_seqs
        # plus 1 since the indices is start from 1 and 0 is remained for padding
        self.n_items = n_items + 1
        self.n_users = n_users + 1
        self.train_test_split = int(train_test_split * len(user_rating_seqs))

    def _get_rating_history_train_batch(self, seq_len, batch_size):
        seqs = np.random.choice(self.user_rating_seqs[:self.train_test_split], batch_size)
        users = [seq[0] for seq in seqs]
        for i in range(len(seqs)):
            start_idx = np.random.randint(1, len(seqs[i]) - seq_len)
            seqs[i] = np.array(seqs[i][start_idx: start_idx + seq_len + 1])
        return np.array(list(seqs)), np.array(users)

    def _get_rating_history_test_batch(self, seq_len, batch_size):
        if batch_size is not None:
            seqs = np.random.choice(self.user_rating_seqs[self.train_test_split:], batch_size)
        else:
            seqs = self.user_rating_seqs[self.train_test_split:]
        users = [seq[0] for seq in seqs]
        for i in range(len(seqs)):
            start_idx = np.random.randint(1, len(seqs[i]) - seq_len)
            seqs[i] = np.array(seqs[i][start_idx: start_idx + seq_len + 1])
        return np.array(list(seqs)), np.array(users)

    def _get_train_batch_from_all_user(self, seq_len, batch_size):
        seqs = np.random.choice(self.user_rating_seqs, batch_size)
        users = [seq[0] for seq in seqs]
        for i in range(len(seqs)):
            start_idx = np.random.randint(1, len(seqs[i]) - 1 - seq_len)  # max_ind: len - seq_len - 2
            seqs[i] = np.array(seqs[i][start_idx: start_idx + seq_len + 1])  # max_ind: len - 1
        return np.array(list(seqs)), np.array(users)

    def _get_test_batch_from_all_user(self, seq_len, batch_size=None):
        if batch_size is not None:
            seqs = np.random.choice(self.user_rating_seqs, batch_size)
        else:
            seqs = self.user_rating_seqs
        users = [seq[0] for seq in seqs]
        seqs = [np.array(seq) for seq in seqs]
        for i in range(len(seqs)):
            seqs[i] = seqs[i][- seq_len - 1:]
        return np.array(list(seqs)), np.array(users)

    def get_train_batch(self, batch_size):
        """
        :param batch_size:
        :return: ratings, ratings, users
            label_mode = 'last':
                [batch_size, seq_len], [batch_size, 1], [batch_size]
            label_mode = 'all'
                [batch_size, seq_len], [batch_size, seq_len], [batch_size]
        """
        if self.train_test_split == 0:
            xs, us = self._get_train_batch_from_all_user(self.seq_len, batch_size)
        else:
            xs, us = self._get_rating_history_train_batch(self.seq_len, batch_size)
        if self.label_mode == "last":
            return xs[:, :-1], xs[:, -1:], us
        else:
            return xs[:, :-1], xs[:, 1:], us

    def get_test_batch(self, batch_size=None):
        """
        :param batch_size:
        :return: ratings, ratings, users
            label_mode = 'last':
                [batch_size, seq_len], [batch_size, 1], [batch_size]
            label_mode = 'all'
                [batch_size, seq_len], [batch_size, seq_len], [batch_size]
        """
        if self.train_test_split == 0:
            xs, us = self._get_test_batch_from_all_user(self.seq_len, batch_size)
        else:
            xs, us = self._get_rating_history_test_batch(self.seq_len, batch_size)
        if self.label_mode == "last":
            return xs[:, :-1], xs[:, -1:], us
        else:
            return xs[:, :-1], xs[:, 1:], us
