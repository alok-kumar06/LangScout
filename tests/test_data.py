import pytest

from langscout.data import chunk_frame, load_clean_dataset, split_rows


@pytest.fixture(scope="module")
def dataset():
    return load_clean_dataset()


def test_label_typos_fixed(dataset):
    df, audit = dataset
    assert {"Portuguese", "Swedish"} <= set(df["language"])
    assert not {"Portugeese", "Sweedish"} & set(df["language"])
    assert audit["languages"] == 17


def test_no_duplicate_or_conflicting_texts(dataset):
    df, _ = dataset
    assert not df["text"].str.lower().duplicated().any()


def test_splits_are_disjoint_and_stratified(dataset):
    df, _ = dataset
    train, val, test = split_rows(df)
    ids = [set(x["row_id"]) for x in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert set(test["language"]) == set(df["language"])           # every language present in test
    texts = [set(x["text"].str.lower()) for x in (train, val, test)]
    assert not (texts[0] & texts[2])                              # no identical text across train/test


def test_chunks_stay_inside_their_split(dataset):
    df, _ = dataset
    train, _, test = split_rows(df)
    assert not (set(chunk_frame(train)["row_id"]) & set(chunk_frame(test)["row_id"]))
