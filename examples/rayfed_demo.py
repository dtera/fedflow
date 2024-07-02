import sys
import ray
import fed
import itertools


@fed.remote
class MyActor:
    def __init__(self, value):
        self.value = value

    def inc(self, num):
        self.value = self.value + num
        return self.value


@fed.remote
def aggregate(val1, val2):
    return val1 + val2


def main(party):
    ray.init(address='local', include_dashboard=False)
    addresses = {
        'alice': '127.0.0.1:10001',
        'bob': '127.0.0.1:10002',
    }
    # Start as alice.
    fed.init(addresses=addresses, party=party)

    actor_alice = MyActor.party("alice").remote(1)
    actor_bob = MyActor.party("bob").remote(1)

    val_alice = actor_alice.inc.remote(1)
    val_bob = actor_bob.inc.remote(2)

    sum_val_obj = aggregate.party("bob").remote(val_alice, val_bob)
    result = fed.get(sum_val_obj)
    print(f"The result in party {party} is {result}")

    fed.shutdown()


if __name__ == "__main__":
    # assert len(sys.argv) == 2, 'Please run this script with party.'
    main(sys.argv[1])
    # stat_test()
