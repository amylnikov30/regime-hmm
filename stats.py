from collections import Counter, defaultdict
from utils import START, END
from math import log

def traverse_corpus(corpus, tag_field="upos", word_field="form", lowercase=False):
    trans_counts = defaultdict(Counter)
    emit_counts = defaultdict(Counter)

    tag_totals = Counter()
    trans_totals = Counter()
    next_totals = Counter()
    
    for sentence in corpus:
        prev = START
        
        for token in sentence:
            tag = token[tag_field]
            word = token[word_field]
            
            if lowercase:
                word = word.lower()
            
            # Count transitions
            trans_counts[prev][tag] += 1
            trans_totals[prev] += 1
            next_totals[tag] += 1
            
            # Count emissions
            emit_counts[tag][word] += 1
            tag_totals[tag] += 1
            
            # Move to next word
            prev = tag
            
        trans_counts[prev][END] += 1
        trans_totals[prev] += 1
        next_totals[END] += 1
        
        # emit_counts[tag][END] += 1
        # tag_totals[tag] += 1
        
    return trans_counts, trans_totals, next_totals, emit_counts, tag_totals

def construct_anchor_map(emit_counts, min_count=2):
    # word -> total count + set of tags it appeared with
    word_totals = Counter()
    word_tags = defaultdict(set)

    for tag, counts in emit_counts.items():
        for word, c in counts.items():
            if word == END:
                continue
            
            word_totals[word] += c
            word_tags[word].add(tag)

    # anchor if appears with exactly one tag and at least twice
    anchor_map = {
        word: next(iter(tags))
        for word, tags in word_tags.items()
        if len(tags) == 1 and word_totals[word] >= min_count
    }

    return anchor_map

def construct_alpha(trans_counts, trans_totals, next_totals, logspace=True):
    alpha = defaultdict(Counter)

    prev_states = list(trans_totals.keys())
    next_states = set(trans_totals.keys())
    next_states.update({END})
    next_states.discard(START)

    for prev in prev_states:
        probs = {}
        for nxt in next_states:
            probs[nxt] = p_wb_bigrams(prev, nxt, trans_counts, trans_totals, next_totals)

        # p_wb_bigrams backs off over trans_totals (which includes START as a key),
        # but START is not a valid destination state. Renormalize on valid columns.
        row_sum = sum(probs.values())
        row = alpha[prev]
        
        inv_row_sum = 1.0 / row_sum
        for nxt, p in probs.items():
            if logspace:
                row[nxt] = log(p * inv_row_sum)
            else:
                row[nxt] = p * inv_row_sum
                
    return alpha

def construct_beta(emit_counts, logspace=True):
    beta = defaultdict(Counter)

    for tag, counts in emit_counts.items():
        row = beta[tag]
        for word in counts.keys():
            row[word] = log(p_wb_unigrams(tag, word, emit_counts)) if logspace else p_wb_unigrams(tag, word, emit_counts)

    return beta

def distinct_followers(w, counts):
    return set(next for prev, count in counts.items() for next, c in count.items() if prev == w and c > 0)

def p_wb_bigrams(w_prev, w, counts, totals, next_totals):
    v = distinct_followers(w_prev, counts)
    
    lam = totals[w_prev] / (totals[w_prev] + len(v))
    
    return lam * counts[w_prev][w] / totals[w_prev] + (1 - lam) * next_totals[w] / sum(next_totals.values())

def p_wb_unigrams(t, w, emit_counts, z=100_000):
    counts = emit_counts[t]
    n = sum(counts.values())
    m = len(counts)
    c = counts.get(w, 0)
    
    if c > 0:
        return c / (n + m)
    
    return m / (z * (n + m))

class Counts:
    def __init__(self, trans_counts, trans_totals, next_totals, emit_counts, tag_totals):
        self.trans_counts = trans_counts
        self.trans_totals = trans_totals
        self.next_totals = next_totals
        self.emit_counts = emit_counts
        self.tag_totals = tag_totals