import sys

import ray
import spu
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal, wait
from secretflow.ml.boost.sgb_v import Sgb
import argparse


def vertical_fed_train(alice, bob, heu, dataset="a9a.train", feat_size_of_label_holder=50, sample_cnt=-1):
    params = {
        'num_boost_round': 3,
        'max_depth': 5,
        # about 20 bin numbers
        'sketch_eps': 0.02,
        'learning_rate': 0.1,
        # use 'linear' if want to do regression
        # for classification, currently only supports binary classfication
        'objective': 'logistic',
        'reg_lambda': 0.3,
        'subsample': 1,
        'colsample_by_tree': 1,
        # pre-pruning parameter. splits with gain value less than it will be pruned.
        'gamma': 1,
    }

    x, y = load_svmlight_file(f"../data/{dataset}")
    x = x.toarray()
    if sample_cnt != -1:
        x = x[:sample_cnt]
        y = y[:sample_cnt]
    # x_guest, y = load_svmlight_file(f"../data/{dataset}.guest.train")
    # x_guest = x_guest.toarray()
    # x_host, _ = load_svmlight_file(f"../data/{dataset}.host.train")
    # x_host = x_host.toarray()

    v_data = FedNdarray(
        {
            alice: (alice(lambda: x[:, :feat_size_of_label_holder])()),
            bob: (bob(lambda: x[:, feat_size_of_label_holder:])()),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    label_data = FedNdarray(
        {alice: (alice(lambda: y)())},
        partition_way=PartitionWay.VERTICAL,
    )
    sgb = Sgb(heu)
    model = sgb.train(params, v_data, label_data)
    yhat = model.predict(v_data)
    yhat = reveal(yhat)
    print(f"auc: {roc_auc_score(y, yhat)}")
    # each participant party needs a location to store
    saving_path_dict = {
        # in production, we may use remote oss, for example.
        device: "./" + device.party
        for device in v_data.partitions.keys()
    }
    r = model.save_model(saving_path_dict)
    wait(r)


def train(p):
    sf.shutdown()
    dataset = p.dataset
    feat_size_of_label_holder = p.feat_size_of_label_holder
    if dataset is not None and "@" in dataset:
        d = dataset.split("@")
        dataset = d[0]
        feat_size_of_label_holder = d[1]
    cluster_config = {
        'parties': {
            'alice': {
                # replace with alice's real address.
                'address': '127.0.0.1:20001',
                'listen_addr': '0.0.0.0:20001'
            },
            'bob': {
                # replace with bob's real address.
                'address': '127.0.0.1:20002',
                'listen_addr': '0.0.0.0:20002'
            },
        },
        'self_party': p.self_party
    }
    _system_config = {'lineage_pinning_enabled': False}
    # address='ray://127.0.0.1:10001'
    if p.self_party is None:
        sf.init(['alice', 'bob'], address='local', _system_config=_system_config, object_store_memory=1024 ** 3)
    else:
        head_addresses = {
            'alice': '127.0.0.1:20001',
            'bob': '127.0.0.1:20002',
        }
        sf.init(address=None, cluster_config=cluster_config)
    # SPU settings
    cluster_def = {
        'nodes': [
            {'party': 'alice', 'address': cluster_config['parties']['alice']['address'].rstrip(':8081') + ':12945'},
            {'party': 'bob', 'address': cluster_config['parties']['bob']['address'].rstrip(':8081') + ':12946'},
        ],
        'runtime_config': {
            # SEMI2K support 2/3 PC, ABY3 only support 3PC, CHEETAH only support 2PC.
            # pls pay attention to size of nodes above. nodes size need match to PC setting.
            'protocol': spu.spu_pb2.SEMI2K,
            'field': spu.spu_pb2.FM128,
            'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
        },
    }
    # HEU settings
    heu_config = {
        'sk_keeper': {'party': 'alice'},
        'evaluators': [{'party': 'bob'}],
        'mode': 'PHEU',
        'he_parameters': {
            # ou is a fast encryption schema that is as secure as paillier.
            'schema': 'ou',
            'key_pair': {
                'generate': {
                    # bit size should be 2048 to provide sufficient security.
                    'bit_size': 2048,
                },
            },
        },
        'encoding': {
            'cleartext_type': 'DT_I32',
            'encoder': "IntegerEncoder",
            'encoder_args': {"scale": 1},
        },
    }
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    heu = sf.HEU(heu_config, cluster_def['runtime_config']['field'])
    vertical_fed_train(alice, bob, heu, dataset, p.feat_size_of_label_holder, p.sample_cnt)
    if p.self_party is not None:
        sf.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='a9a.train', help="a9a.train.")
    parser.add_argument("--self_party", type=str, default=None, help="self_party.")
    parser.add_argument("--feat_size_of_label_holder", type=int, default=50, help="feat size of guest.")
    parser.add_argument("--sample_cnt", type=int, default=-1, help="sample_cnt.")
    p = parser.parse_args()
    train(p)
