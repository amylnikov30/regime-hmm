from collections import Counter, defaultdict
from math import log

from stats import p_wb_unigrams
from logsumexptrick import logsumexp, min_log_prob
from utils import START, END

def get_emit_prob(q, w, beta, emit_counts, logspace=True, anchors=None, use_anchors=False):
    # Anchor logic
    if use_anchors and anchors:
        forced_tag = anchors.get(w)
        if forced_tag is not None:
            if q == forced_tag:
                return 0.0 if logspace else 1.0
            else:
                return min_log_prob if logspace else 0.0

    # p_wb smoothing logic
    row = beta[q]
    if w in row:
        return row[w]

    p = p_wb_unigrams(q, w, emit_counts)
    return log(p) if logspace else p


def construct_model(
    sent,
    alpha,
    beta,
    emit_counts,
    logspace=True,
    anchors=None,
    use_anchors=False,
    word_field="form",
):
    viterbi = defaultdict(Counter)
    backpointers = defaultdict(Counter)

    forward = defaultdict(Counter)

    # Handle start state
    w0 = sent[0][word_field]
    alpha_start = alpha[START]

    for q, start_prob in alpha_start.items():
        if q == END:
            continue

        b = get_emit_prob(
            q,
            w0,
            beta,
            emit_counts,
            logspace=logspace,
            anchors=anchors,
            use_anchors=use_anchors,
        )
        if logspace:
            viterbi[q][0] = start_prob + b
            forward[q][0] = start_prob + b
        else:
            viterbi[q][0] = start_prob * b
            forward[q][0] = start_prob * b

        backpointers[q][0] = START

    # Main iteration
    for i in range(1, len(sent)):
        w = sent[i][word_field]

        for q in viterbi.keys():
            max, argmax = max_q(
                viterbi,
                alpha,
                beta,
                emit_counts,
                q,
                i,
                w,
                logspace=logspace,
                anchors=anchors,
                use_anchors=use_anchors,
            )

            viterbi[q][i] = max
            backpointers[q][i] = argmax

            forward[q][i] = forward_sum(
                forward,
                alpha,
                beta,
                emit_counts,
                q,
                i,
                w,
                logspace=logspace,
                anchors=anchors,
                use_anchors=use_anchors,
            )

    # Handle final state
    beta_1 = defaultdict(Counter)
    beta_1[END][""] = 0.0

    emit_counts_1 = defaultdict(Counter)
    emit_counts_1[END][""] = 1.0

    max_f, argmax_f = max_q(
        viterbi,
        alpha,
        beta_1,
        emit_counts_1,
        END,
        len(sent),
        "",
        logspace=logspace,
        anchors=anchors,
        use_anchors=use_anchors,
    )

    viterbi[END][len(sent) - 1] = max_f
    forward[END][len(sent) - 1] = forward_sum(
        forward,
        alpha,
        beta_1,
        emit_counts_1,
        END,
        len(sent),
        "",
        logspace=logspace,
        anchors=anchors,
        use_anchors=use_anchors,
    )

    # Backtrack to get path
    path = [argmax_f]

    for i in range(len(sent) - 1, 0, -1):
        q = path[-1]

        path.append(backpointers[q][i])

    return path[::-1], viterbi, backpointers, forward


def forward_sum(
    forward,
    alpha,
    beta,
    emit_counts,
    q,
    i,
    w,
    logspace=True,
    anchors=None,
    use_anchors=False,
):
    weights = []

    for q_prev in forward.keys():
        v = forward[q_prev][i - 1]
        a = alpha[q_prev][q]
        b = get_emit_prob(
            q,
            w,
            beta,
            emit_counts,
            logspace=logspace,
            anchors=anchors,
            use_anchors=use_anchors,
        )

        if logspace:
            weight = v + a + b
        else:
            weight = v * a * b

        weights.append(weight)

    if logspace:
        return logsumexp(weights)

    return sum(weights)


def max_q(
    viterbi,
    alpha,
    beta,
    emit_counts,
    q,
    i,
    w,
    logspace=True,
    anchors=None,
    use_anchors=False,
):
    weights = []

    for q_prev in viterbi.keys():
        v = viterbi[q_prev][i - 1]
        a = alpha[q_prev][q]
        b = get_emit_prob(
            q,
            w,
            beta,
            emit_counts,
            logspace=logspace,
            anchors=anchors,
            use_anchors=use_anchors,
        )

        if logspace:
            weight = v + a + b
        else:
            weight = v * a * b

        weights.append((weight, q_prev))

    result = max(weights, key=lambda x: x[0])

    return result[0], result[1]
