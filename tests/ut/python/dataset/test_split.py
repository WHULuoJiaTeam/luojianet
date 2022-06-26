# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import pytest
import mindspore.dataset as ds
from util import config_get_set_num_parallel_workers


# test5trainimgs.json contains 5 images whose un-decoded shape is [83554, 54214, 65512, 54214, 64631]
# the label of each image is [0,0,0,1,1] each image can be uniquely identified
# via the following lookup table (dict){(83554, 0): 0, (54214, 0): 1, (54214, 1): 2, (65512, 0): 3, (64631, 1): 4}
manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"
manifest_map = {(172876, 0): 0, (54214, 0): 1, (54214, 1): 2, (173673, 0): 3, (64631, 1): 4}

text_file_dataset_path = "../data/dataset/testTextFileDataset/*"
text_file_data = ["This is a text file.", "Another file.", "Be happy every day.",
                  "End of file.", "Good luck to everyone."]

def split_with_invalid_inputs(d):
    with pytest.raises(ValueError) as info:
        _, _ = d.split([])
    assert "sizes cannot be empty" in str(info.value)

    with pytest.raises(ValueError) as info:
        _, _ = d.split([5, 0.6])
    assert "sizes should be list of int or list of float" in str(info.value)

    with pytest.raises(ValueError) as info:
        _, _ = d.split([-1, 6])
    assert "there should be no negative or zero numbers" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        _, _ = d.split([3, 1])
    assert "Sum of split sizes 4 is not equal to dataset size 5" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        _, _ = d.split([5, 1])
    assert "Sum of split sizes 6 is not equal to dataset size 5" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        _, _ = d.split([0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
    assert "Sum of calculated split sizes 6 is not equal to dataset size 5" in str(info.value)

    with pytest.raises(ValueError) as info:
        _, _ = d.split([-0.5, 0.5])
    assert "there should be no numbers outside the range (0, 1]" in str(info.value)

    with pytest.raises(ValueError) as info:
        _, _ = d.split([1.5, 0.5])
    assert "there should be no numbers outside the range (0, 1]" in str(info.value)

    with pytest.raises(ValueError) as info:
        _, _ = d.split([0.5, 0.6])
    assert "percentages do not sum up to 1" in str(info.value)

    with pytest.raises(ValueError) as info:
        _, _ = d.split([0.3, 0.6])
    assert "percentages do not sum up to 1" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        _, _ = d.split([0.05, 0.95])
    assert "percentage 0.05 is too small" in str(info.value)


def test_unmappable_invalid_input():
    d = ds.TextFileDataset(text_file_dataset_path)
    split_with_invalid_inputs(d)

    d = ds.TextFileDataset(text_file_dataset_path, num_shards=2, shard_id=0)
    with pytest.raises(RuntimeError) as info:
        _, _ = d.split([4, 1])
    assert "Dataset should not be sharded before split" in str(info.value)


def test_unmappable_split():
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)

    d = ds.TextFileDataset(text_file_dataset_path, shuffle=False)
    s1, s2 = d.split([4, 1], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(item["text"].item().decode("utf8"))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(item["text"].item().decode("utf8"))

    assert s1_output == text_file_data[0:4]
    assert s2_output == text_file_data[4:]

    # exact percentages
    s1, s2 = d.split([0.8, 0.2], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(item["text"].item().decode("utf8"))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(item["text"].item().decode("utf8"))

    assert s1_output == text_file_data[0:4]
    assert s2_output == text_file_data[4:]

    # fuzzy percentages
    s1, s2 = d.split([0.33, 0.67], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(item["text"].item().decode("utf8"))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(item["text"].item().decode("utf8"))

    assert s1_output == text_file_data[0:2]
    assert s2_output == text_file_data[2:]

    # Restore configuration num_parallel_workers
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_unmappable_randomize_deterministic():
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)

    # the labels outputted by ShuffleOp for seed 53 is [0, 2, 1, 4, 3]
    ds.config.set_seed(53)

    d = ds.TextFileDataset(text_file_dataset_path, shuffle=False)
    s1, s2 = d.split([0.8, 0.2])

    for _ in range(10):
        s1_output = []
        for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
            s1_output.append(item["text"].item().decode("utf8"))

        s2_output = []
        for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
            s2_output.append(item["text"].item().decode("utf8"))

        # note no overlap
        assert s1_output == [text_file_data[0], text_file_data[2], text_file_data[1], text_file_data[4]]
        assert s2_output == [text_file_data[3]]

    # Restore configuration num_parallel_workers
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_unmappable_randomize_repeatable():
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)

    # the labels outputted by ShuffleOp for seed 53 is [0, 2, 1, 4, 3]
    ds.config.set_seed(53)

    d = ds.TextFileDataset(text_file_dataset_path, shuffle=False)
    s1, s2 = d.split([0.8, 0.2])

    num_epochs = 5
    s1 = s1.repeat(num_epochs)
    s2 = s2.repeat(num_epochs)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(item["text"].item().decode("utf8"))

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(item["text"].item().decode("utf8"))

    # note no overlap
    assert s1_output == [text_file_data[0], text_file_data[2], text_file_data[1], text_file_data[4]] * num_epochs
    assert s2_output == [text_file_data[3]] * num_epochs

    # Restore configuration num_parallel_workers
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_unmappable_get_dataset_size():
    d = ds.TextFileDataset(text_file_dataset_path, shuffle=False)
    s1, s2 = d.split([0.8, 0.2])

    assert d.get_dataset_size() == 5
    assert s1.get_dataset_size() == 4
    assert s2.get_dataset_size() == 1


def test_unmappable_multi_split():
    original_num_parallel_workers = config_get_set_num_parallel_workers(4)

    # the labels outputted by ShuffleOp for seed 53 is [0, 2, 1, 4, 3]
    ds.config.set_seed(53)

    d = ds.TextFileDataset(text_file_dataset_path, shuffle=False)
    s1, s2 = d.split([4, 1])

    s1_correct_output = [text_file_data[0], text_file_data[2], text_file_data[1], text_file_data[4]]

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(item["text"].item().decode("utf8"))
    assert s1_output == s1_correct_output

    # no randomize in second split
    s1s1, s1s2, s1s3 = s1.split([1, 2, 1], randomize=False)

    s1s1_output = []
    for item in s1s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s1_output.append(item["text"].item().decode("utf8"))

    s1s2_output = []
    for item in s1s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s2_output.append(item["text"].item().decode("utf8"))

    s1s3_output = []
    for item in s1s3.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s3_output.append(item["text"].item().decode("utf8"))

    assert s1s1_output == [s1_correct_output[0]]
    assert s1s2_output == [s1_correct_output[1], s1_correct_output[2]]
    assert s1s3_output == [s1_correct_output[3]]

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(item["text"].item().decode("utf8"))
    assert s2_output == [text_file_data[3]]

    # randomize in second split
    # the labels outputted by the ShuffleOp for seed 53 is [2, 3, 1, 0]
    shuffled_ids = [2, 3, 1, 0]

    s1s1, s1s2, s1s3 = s1.split([1, 2, 1])

    s1s1_output = []
    for item in s1s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s1_output.append(item["text"].item().decode("utf8"))

    s1s2_output = []
    for item in s1s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s2_output.append(item["text"].item().decode("utf8"))

    s1s3_output = []
    for item in s1s3.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s3_output.append(item["text"].item().decode("utf8"))

    assert s1s1_output == [s1_correct_output[shuffled_ids[0]]]
    assert s1s2_output == [s1_correct_output[shuffled_ids[1]], s1_correct_output[shuffled_ids[2]]]
    assert s1s3_output == [s1_correct_output[shuffled_ids[3]]]

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(item["text"].item().decode("utf8"))
    assert s2_output == [text_file_data[3]]

    # Restore configuration num_parallel_workers
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_mappable_invalid_input():
    d = ds.ManifestDataset(manifest_file)
    split_with_invalid_inputs(d)

    d = ds.ManifestDataset(manifest_file, num_shards=2, shard_id=0)
    with pytest.raises(RuntimeError) as info:
        _, _ = d.split([4, 1])
    assert "Dataset should not be sharded before split" in str(info.value)


def test_mappable_split_general():
    d = ds.ManifestDataset(manifest_file, shuffle=False)
    d = d.take(5)

    # absolute rows
    s1, s2 = d.split([4, 1], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    assert s1_output == [0, 1, 2, 3]
    assert s2_output == [4]

    # exact percentages
    s1, s2 = d.split([0.8, 0.2], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    assert s1_output == [0, 1, 2, 3]
    assert s2_output == [4]

    # fuzzy percentages
    s1, s2 = d.split([0.33, 0.67], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    assert s1_output == [0, 1]
    assert s2_output == [2, 3, 4]


def test_mappable_split_optimized():
    d = ds.ManifestDataset(manifest_file, shuffle=False)

    # absolute rows
    s1, s2 = d.split([4, 1], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    assert s1_output == [0, 1, 2, 3]
    assert s2_output == [4]

    # exact percentages
    s1, s2 = d.split([0.8, 0.2], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    assert s1_output == [0, 1, 2, 3]
    assert s2_output == [4]

    # fuzzy percentages
    s1, s2 = d.split([0.33, 0.67], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    assert s1_output == [0, 1]
    assert s2_output == [2, 3, 4]


def test_mappable_randomize_deterministic():
    # the labels outputted by ManifestDataset for seed 53 is [0, 1, 3, 4, 2]
    ds.config.set_seed(53)

    d = ds.ManifestDataset(manifest_file, shuffle=False)
    s1, s2 = d.split([0.8, 0.2])

    for _ in range(10):
        s1_output = []
        for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
            s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

        s2_output = []
        for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
            s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

        # note no overlap
        assert s1_output == [0, 1, 3, 4]
        assert s2_output == [2]


def test_mappable_randomize_repeatable():
    # the labels outputted by ManifestDataset for seed 53 is [0, 1, 3, 4, 2]
    ds.config.set_seed(53)

    d = ds.ManifestDataset(manifest_file, shuffle=False)
    s1, s2 = d.split([0.8, 0.2])

    num_epochs = 5
    s1 = s1.repeat(num_epochs)
    s2 = s2.repeat(num_epochs)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    # note no overlap
    assert s1_output == [0, 1, 3, 4] * num_epochs
    assert s2_output == [2] * num_epochs


def test_mappable_sharding():
    # set arbitrary seed for repeatability for shard after split
    # the labels outputted by ManifestDataset for seed 53 is [0, 1, 3, 4, 2]
    ds.config.set_seed(53)

    num_epochs = 5
    first_split_num_rows = 4

    d = ds.ManifestDataset(manifest_file, shuffle=False)
    s1, s2 = d.split([first_split_num_rows, 1])

    distributed_sampler = ds.DistributedSampler(2, 0)
    s1.use_sampler(distributed_sampler)

    s1 = s1.repeat(num_epochs)

    # testing sharding, second dataset to simulate another instance
    d2 = ds.ManifestDataset(manifest_file, shuffle=False)
    d2s1, d2s2 = d2.split([first_split_num_rows, 1])

    distributed_sampler = ds.DistributedSampler(2, 1)
    d2s1.use_sampler(distributed_sampler)

    d2s1 = d2s1.repeat(num_epochs)

    # shard 0
    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    # shard 1
    d2s1_output = []
    for item in d2s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        d2s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    rows_per_shard_per_epoch = 2
    assert len(s1_output) == rows_per_shard_per_epoch * num_epochs
    assert len(d2s1_output) == rows_per_shard_per_epoch * num_epochs

    # verify each epoch that
    #   1. shards contain no common elements
    #   2. the data was split the same way, and that the union of shards equal the split
    correct_sorted_split_result = [0, 1, 3, 4]
    for i in range(num_epochs):
        combined_data = []
        for j in range(rows_per_shard_per_epoch):
            combined_data.append(s1_output[i * rows_per_shard_per_epoch + j])
            combined_data.append(d2s1_output[i * rows_per_shard_per_epoch + j])

        assert sorted(combined_data) == correct_sorted_split_result

    # test other split
    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    d2s2_output = []
    for item in d2s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        d2s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    assert s2_output == [2]
    assert d2s2_output == [2]


def test_mappable_get_dataset_size():
    d = ds.ManifestDataset(manifest_file, shuffle=False)
    s1, s2 = d.split([4, 1])

    assert d.get_dataset_size() == 5
    assert s1.get_dataset_size() == 4
    assert s2.get_dataset_size() == 1


def test_mappable_multi_split():
    # the labels outputted by ManifestDataset for seed 53 is [0, 1, 3, 4, 2]
    ds.config.set_seed(53)

    d = ds.ManifestDataset(manifest_file, shuffle=False)
    s1, s2 = d.split([4, 1])

    s1_correct_output = [0, 1, 3, 4]

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])
    assert s1_output == s1_correct_output

    # no randomize in second split
    s1s1, s1s2, s1s3 = s1.split([1, 2, 1], randomize=False)

    s1s1_output = []
    for item in s1s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s1s2_output = []
    for item in s1s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s1s3_output = []
    for item in s1s3.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s3_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    assert s1s1_output == [s1_correct_output[0]]
    assert s1s2_output == [s1_correct_output[1], s1_correct_output[2]]
    assert s1s3_output == [s1_correct_output[3]]

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])
    assert s2_output == [2]

    # randomize in second split
    # the labels outputted by the RandomSampler for seed 53 is [3, 1, 2, 0]
    random_sampler_ids = [3, 1, 2, 0]

    s1s1, s1s2, s1s3 = s1.split([1, 2, 1])

    s1s1_output = []
    for item in s1s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s1s2_output = []
    for item in s1s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s1s3_output = []
    for item in s1s3.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1s3_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    assert s1s1_output == [s1_correct_output[random_sampler_ids[0]]]
    assert s1s2_output == [s1_correct_output[random_sampler_ids[1]], s1_correct_output[random_sampler_ids[2]]]
    assert s1s3_output == [s1_correct_output[random_sampler_ids[3]]]

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])
    assert s2_output == [2]


def test_rounding():
    d = ds.ManifestDataset(manifest_file, shuffle=False)

    # under rounding
    s1, s2 = d.split([0.5, 0.5], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    assert s1_output == [0, 1, 2]
    assert s2_output == [3, 4]

    # over rounding
    s1, s2, s3 = d.split([0.15, 0.55, 0.3], randomize=False)

    s1_output = []
    for item in s1.create_dict_iterator(num_epochs=1, output_numpy=True):
        s1_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s2_output = []
    for item in s2.create_dict_iterator(num_epochs=1, output_numpy=True):
        s2_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    s3_output = []
    for item in s3.create_dict_iterator(num_epochs=1, output_numpy=True):
        s3_output.append(manifest_map[(item["image"].shape[0], item["label"].item())])

    assert s1_output == [0]
    assert s2_output == [1, 2]
    assert s3_output == [3, 4]


if __name__ == '__main__':
    test_unmappable_invalid_input()
    test_unmappable_split()
    test_unmappable_randomize_deterministic()
    test_unmappable_randomize_repeatable()
    test_unmappable_get_dataset_size()
    test_unmappable_multi_split()
    test_mappable_invalid_input()
    test_mappable_split_general()
    test_mappable_split_optimized()
    test_mappable_randomize_deterministic()
    test_mappable_randomize_repeatable()
    test_mappable_sharding()
    test_mappable_get_dataset_size()
    test_mappable_multi_split()
    test_rounding()
