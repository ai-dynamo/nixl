### Elastic Test Suite

#### Single Node (8 ranks, 4→8 expansion):
```bash
python3 tests/elastic/elastic.py \
    --plan tests/elastic/single_expansion.json \
    --num-processes 8 \
    --etcd-server http://127.0.0.1:2379
```

#### Multi-Node Setup:

**Node 1** (will launch the first phase with 4 ranks):
```bash
python3 tests/elastic/elastic.py \
    --plan tests/elastic/single_expansion.json \
    --num-processes 4 \
```

**Node 2** (will join the second phase with additional 4 ranks):
```bash
python3 tests/elastic/elastic.py \
    --plan tests/elastic/single_expansion.json \
    --num-processes 4 \
    --rank-server $MASTER_IP \
    --etcd-server http://$MASTER_IP:2379
```

### Available Test Plans

- `no_expansion.json`: Static 4 ranks (baseline)
- `single_expansion.json`: 4 → 8 ranks (single expansion)
- `double_expansion.json`: 4 → 6 → 8 ranks (two expansions)
- `expansion_contraction.json`: 4 → 8 → 6 ranks (scale up then down)
