import os
import gzip


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


src_bpe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dlc", "bpe_simple_vocab_16e6.txt.gz")

merges = gzip.open(src_bpe).read().decode("utf-8").split('\n')
merges = merges[1:49152-256-2+1]
merges = [tuple(merge.split()) for merge in merges]
assert all(len(m) == 2 and ' ' not in m[0] and ' ' not in m[1] and '\n' not in m[0] and '\n' not in m[1] for m in merges)

vocab = list(bytes_to_unicode().values())
vocab = vocab + [v+'</w>' for v in vocab]
assert all(' ' not in v and '\n' not in v for v in vocab)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dlc', 'ctokenizer.txt'), 'wb') as f:
    for v in vocab:
        f.write((v + '\n').encode('utf-8'))
    for m in merges:
        f.write((m[0] + ' ' + m[1] + '\n').encode('utf-8'))
