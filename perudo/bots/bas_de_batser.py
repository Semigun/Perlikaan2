"""
Bas de Batser -- a static (non-learning) Bayesian Perudo bot.

Same engine as Orakel (hand posteriors, expected-value choice between
Dudo, Calza and bids, continuation risk), but with ALL online learning
switched off: it plays every game with the same fixed assumptions about
opponents (they mostly bid faces they hold; they call Dudo when a bid
looks less than ~20% likely).  Its signature quirk: whenever
it has to open a round, it opens with one 2 ("1x 2"), whatever its dice.

Self-contained: standard library + perudo.models only.  Never raises.
"""

from __future__ import annotations

import math
import random

from perudo.models import (
    PACO,
    Bid,
    BidEvent,
    Calza,
    CalzaEvent,
    DieGainedEvent,
    DieLostEvent,
    Dudo,
    DudoEvent,
    PlayerEliminatedEvent,
    RevealEvent,
    RoundEndedEvent,
)

MAX_DICE = 5

# ---------------------------------------------------------------------------
# Tunable parameters (found by simulation, see research/)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "gamma": 1.5,          # value model exponent: W = d^g / sum d^g
    "kappa": 0.8,          # multiplier on the continuation-risk prior
    "rb_m": 20.0,          # pseudo-counts of that prior vs. own experience
    "learn_rback": False,  # no learning: fixed continuation-risk prior
    "max_steps": 4,        # candidate quantities per face above the minimum
    "alpha_bid": 6.0,      # per-opponent smoothing towards the population
    "alpha_pop": 30.0,     # population smoothing towards "no information"
    "beta_call": 4.0,      # per-opponent smoothing of the calling model
    "beta_pop": 20.0,      # population smoothing of the calling model
    "learn": False,        # no learning: fixed assumptions only
    "use_evidence": True,  # use opponents' bids as evidence about hands
    "calza_ok": True,
    "use_excess": True,    # condition bid evidence on how big the jump was
    "accept_pow": 0.0,     # weight of "didn't call Dudo" as evidence (0 = off)
    "w_prior_n": 20.0,     # weight of population evidence on naive-vs-bayes
    "memory": 3000,        # per-opponent decisions remembered before old
                           # data is halved (tracks opponents that adapt)
    "explore_eps": 0.0,    # pick randomly among bids within eps of the best
                           # EV, so learning opponents can't read my hand
    "seed": 20261003,
    "honesty_beta": 0.3,   # fixed belief: bids tilt hands towards the bid face
    "call_center": 0.2,    # fixed belief: opponents call below this P(true)
    "strict": False,       # re-raise internal errors (tests only)
}

# ---------------------------------------------------------------------------
# Combinatorics tables
# ---------------------------------------------------------------------------


def _enumerate_hands(n):
    """All multisets of n dice as count vectors (c1..c6) with their probability."""
    out = []

    def rec(face, left, counts):
        if face == 6:
            counts.append(left)
            ways = math.factorial(n)
            for c in counts:
                ways //= math.factorial(c)
            out.append((ways / 6.0 ** n, tuple(counts)))
            counts.pop()
            return
        for c in range(left + 1):
            counts.append(c)
            rec(face + 1, left - c, counts)
            counts.pop()

    rec(1, n, [])
    return out


HANDS = {n: _enumerate_hands(n) for n in range(0, MAX_DICE + 1)}
HAND_PRIOR = {n: [p for p, _ in HANDS[n]] for n in HANDS}


def _match(counts, face, palifico):
    if palifico or face == PACO:
        return counts[face - 1]
    return counts[face - 1] + counts[0]


# MATCH[n][palifico][face] -> list of match counts, aligned with HANDS[n]
MATCH = {
    n: {
        pal: {f: [_match(c, f, pal) for _, c in HANDS[n]] for f in range(1, 7)}
        for pal in (False, True)
    }
    for n in HANDS
}


def _binom_pmf(n, p):
    return [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]


# BINOM[n][p6] where the per-die match probability is p6/6
BINOM = {n: {1: _binom_pmf(n, 1 / 6), 2: _binom_pmf(n, 2 / 6)} for n in range(0, 61)}


def _p6(face, palifico):
    return 1 if (palifico or face == PACO) else 2


def _binom(n, p6):
    if n not in BINOM:
        BINOM[n] = {1: _binom_pmf(n, 1 / 6), 2: _binom_pmf(n, 2 / 6)}
    return BINOM[n][p6]


_TAIL_CACHE = {}


def _binom_tail(n, need, p6):
    """P(Binomial(n, p6/6) >= need)."""
    if need <= 0:
        return 1.0
    if need > n:
        return 0.0
    key = (n, need, p6)
    v = _TAIL_CACHE.get(key)
    if v is None:
        v = sum(_binom(n, p6)[need:])
        _TAIL_CACHE[key] = v
    return v


def _conv(a, b):
    out = [0.0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0.0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _tail(dist, need):
    """P(X >= need) for a pmf list."""
    if need <= 0:
        return 1.0
    if need >= len(dist):
        return 0.0
    return sum(dist[need:])


def _sigmoid(x):
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


CALL_BINS = 20


def _call_bin(x):
    b = int(x * CALL_BINS)
    return min(max(b, 0), CALL_BINS - 1)


def _prior_dudo(x, center=0.4):
    # Fixed assumption: bots call when a bid looks less likely than `center`.
    return 0.02 + 0.9 * _sigmoid(12.0 * (center - x))


# Continuation risk: P(I still lose a die this round | my bid was NOT
# challenged by the next player), by players alive and the bid's public
# plausibility psi.  Prior shape measured in simulation (research/), then
# refined online from the bot's own experience.
RB_PSI_BINS = 10
_RB_C = {2: 0.5, 3: 0.65, 4: 0.45, 5: 0.3, 6: 0.25}
_RB_E = {2: 0.6, 3: 1.5, 4: 2.0, 5: 2.5, 6: 3.5}


def _rb_n(n_alive):
    return min(n_alive, 7)


def _rb_prior(n_alive, psi):
    n = _rb_n(n_alive)
    c = _RB_C.get(n, 0.2)
    e = _RB_E.get(n, 4.0)
    return min(0.95, c * psi ** e + (0.1 if n == 2 else 0.0))


def _rb_bin(psi):
    return min(RB_PSI_BINS - 1, max(0, int(psi * RB_PSI_BINS)))


# ---------------------------------------------------------------------------
# Opponent / population statistics
# ---------------------------------------------------------------------------


class _CallBins:
    """Counts of decisions facing a bid, binned by a P(bid true) feature."""

    def __init__(self):
        self.n = [0] * CALL_BINS
        self.d = [0] * CALL_BINS
        self.c = [0] * CALL_BINS

    def add(self, x, outcome):
        b = _call_bin(x)
        self.n[b] += 1
        if outcome == "dudo":
            self.d[b] += 1
        elif outcome == "calza":
            self.c[b] += 1


class _Stats:
    """Learned behaviour of one opponent (or of the whole population)."""

    def __init__(self):
        # (bid_type, p6, n) -> counts of k (#matches of the bid face in hand)
        self.bid = {}
        # Calling behaviour under two hypotheses about how they think:
        #   naive: P(bid true) from own dice + binomial for everyone else
        #   bayes: same, but reading the other bidders' hands from their bids
        self.call_naive = _CallBins()
        self.call_bayes = _CallBins()
        # prequential log-likelihood of each hypothesis (model averaging)
        self.ll = [0.0, 0.0]
        self.n_calls = 0
        self.rounds = 0

    def add_bid(self, key, n, k):
        cnt = self.bid.get(key)
        if cnt is None:
            cnt = self.bid[key] = [0] * (n + 1)
        cnt[k] += 1

    def halve(self):
        """Forget half of everything: old behaviour fades, new dominates."""
        for cnt in self.bid.values():
            for i in range(len(cnt)):
                cnt[i] *= 0.5
        for bins in (self.call_naive, self.call_bayes):
            for arr in (bins.n, bins.d, bins.c):
                for i in range(len(arr)):
                    arr[i] *= 0.5
        self.ll = [x * 0.5 for x in self.ll]
        self.n_calls *= 0.5


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------


class BasDeBatserBot:
    name = "Bas de Batser"
    count = 1

    def __init__(self, **params):
        self.p = dict(DEFAULTS)
        self.p.update(params)
        self.stats = {}            # opponent name -> _Stats
        self.pop = _Stats()        # pooled over all opponents
        self._lr_cache = {}
        self._call_cache = {}
        self.rb = {}               # (n_alive, psi_bin) -> [lost, total]
        self.rng = random.Random(self.p["seed"])
        self._reset_game()

    # -- lifecycle ---------------------------------------------------------

    def _reset_game(self):
        self.dice = {}             # name -> dice count (tracked from events)
        self.round_log = []        # (player, action, prev_bid, palifico, total)
        self.cur_bid = None
        self.palifico = False
        self.post_cache = {}
        self.my_bids = []          # (n_alive, psi_bin) of my bids this round

    def on_game_start(self, info):
        self._reset_game()
        for p in info.players:
            self.dice[p.name] = p.dice_count

    def on_game_end(self, result):
        pass

    # -- observing (learning) ---------------------------------------------

    def observe(self, event):
        try:
            self._observe(event)
        except Exception:
            # Learning must never be able to crash the game.
            if self.p["strict"]:
                raise
            self.round_log = []
            self.cur_bid = None

    def _observe(self, event):
        if isinstance(event, BidEvent):
            self.round_log.append(
                (event.player, event.bid, self.cur_bid, self.palifico, sum(self.dice.values()))
            )
            self.cur_bid = event.bid
        elif isinstance(event, DudoEvent):
            self.round_log.append(
                (event.player, "dudo", self.cur_bid, self.palifico, sum(self.dice.values()))
            )
        elif isinstance(event, CalzaEvent):
            self.round_log.append(
                (event.player, "calza", self.cur_bid, self.palifico, sum(self.dice.values()))
            )
        elif isinstance(event, RevealEvent):
            if self.p["learn"]:
                self._learn_from_reveal(event)
            if self.p["learn_rback"]:
                self._learn_rback(event)
            self.my_bids = []
            self.round_log = []
            self.cur_bid = None
        elif isinstance(event, DieLostEvent) or isinstance(event, DieGainedEvent):
            self.dice[event.player] = event.dice_count
        elif isinstance(event, PlayerEliminatedEvent):
            self.dice[event.player] = 0
        elif isinstance(event, RoundEndedEvent):
            self.palifico = event.palifico_next
            self.round_log = []
            self.cur_bid = None
            self.post_cache = {}
            self.my_bids = []

    def _learn_rback(self, ev):
        if not self.my_bids:
            return
        me = self.name
        if ev.challenge_type == "dudo":
            loser = ev.challenger if ev.result == "bidder_correct" else ev.bidder
        else:
            loser = ev.challenger if ev.result == "wrong" else None
        last = len(self.my_bids) - 1
        for i, key in enumerate(self.my_bids):
            if i == last and ev.bidder == me:
                continue  # challenged straight away: not a continuation
            c = self.rb.get(key)
            if c is None:
                c = self.rb[key] = [0, 0]
            c[0] += loser == me
            c[1] += 1

    def _r_back(self, n_alive, psi):
        prior = min(0.95, self.p["kappa"] * _rb_prior(n_alive, psi))
        c = self.rb.get((_rb_n(n_alive), _rb_bin(psi)))
        if not c:
            return prior
        m = self.p["rb_m"]
        return (c[0] + m * prior) / (c[1] + m)

    def _opp_stats(self, name):
        s = self.stats.get(name)
        if s is None:
            s = self.stats[name] = _Stats()
        return s

    def _learn_from_reveal(self, ev):
        hands = ev.dice_by_player()
        me = self.name
        seen_faces = set()
        counted_round = set()
        sizes = {name: len(h) for name, h in hands.items() if h}
        evidence = {}  # what the table could infer so far, per player
        cm_cache = {}

        def cm(name):
            v = cm_cache.get(name)
            if v is None:
                v = cm_cache[name] = self._call_model(name)
            return v

        for player, action, prev, pal, total in self.round_log:
            hand = hands.get(player)
            if not hand:
                continue
            n = len(hand)
            st = self._opp_stats(player)
            if player not in counted_round:
                counted_round.add(player)
                st.rounds += 1
            if prev is not None and player != me:
                # How believable was `prev` from this player's seat?
                p6 = _p6(prev.face, pal)
                k = _count(hand, prev.face, pal)
                xn = _binom_tail(total - n, prev.quantity - k, p6)
                dist = [1.0]
                for other, m in sizes.items():
                    if other != player:
                        dist = _conv(dist, self._player_dist(other, m, pal, evidence.get(other), prev.face))
                xb = _tail(dist, prev.quantity - k)
                outcome = action if isinstance(action, str) else "bid"
                called = outcome == "dudo"
                for who, stats in ((player, st), (None, self.pop)):
                    naive, bayes, _ = cm(who)
                    pn = min(0.999, max(0.001, naive[_call_bin(xn)][0]))
                    pb = min(0.999, max(0.001, bayes[_call_bin(xb)][0]))
                    stats.ll[0] += math.log(pn if called else 1.0 - pn)
                    stats.ll[1] += math.log(pb if called else 1.0 - pb)
                    stats.call_naive.add(xn, outcome)
                    stats.call_bayes.add(xb, outcome)
                    stats.n_calls += 1
            if isinstance(action, Bid):
                key_face = (player, action.face)
                if key_face in seen_faces:
                    continue
                seen_faces.add(key_face)
                t = _bid_type(action, prev, total, pal, self.p["use_excess"])
                p6 = _p6(action.face, pal)
                if n <= MAX_DICE:
                    evidence.setdefault(player, []).append(
                        (action.face, self._likelihood_ratio(player, t, p6, n), ("bid", action.face, t))
                    )
                k = _count(hand, action.face, pal)
                key = (t, p6, n)
                st.add_bid(key, n, k)
                if player != me:
                    self.pop.add_bid(key, n, k)
        cap = self.p["memory"]
        if cap:
            for name in counted_round:
                st = self.stats.get(name)
                if st is not None and st.n_calls > cap:
                    st.halve()
            if self.pop.n_calls > 6 * cap:
                self.pop.halve()
        self._lr_cache = {}
        self._call_cache = {}

    # -- learned models ----------------------------------------------------

    def _likelihood_ratio(self, name, t, p6, n):
        """LR over k: P(k | this player made such a bid) / P(k) a priori."""
        key = (name, t, p6, n)
        lr = self._lr_cache.get(key)
        if lr is not None:
            return lr
        prior = _binom(n, p6)
        pop_cnt = self.pop.bid.get((t, p6, n))
        a0 = self.p["alpha_pop"]
        if pop_cnt:
            tot = sum(pop_cnt)
            pop = [(pop_cnt[k] + a0 * prior[k]) / (tot + a0) for k in range(n + 1)]
        else:
            # Fixed assumption: bidders tend to hold the face they bid.
            beta = self.p["honesty_beta"]
            tilt = [prior[k] * math.exp(beta * k) for k in range(n + 1)]
            z = sum(tilt)
            pop = [x / z for x in tilt]
        st = self.stats.get(name)
        cnt = st.bid.get((t, p6, n)) if st else None
        a = self.p["alpha_bid"]
        if cnt:
            tot = sum(cnt)
            post = [(cnt[k] + a * pop[k]) / (tot + a) for k in range(n + 1)]
        else:
            post = pop
        lr = [post[k] / prior[k] if prior[k] > 0 else 1.0 for k in range(n + 1)]
        self._lr_cache[key] = lr
        return lr

    def _call_model(self, name):
        """(naive table, bayes table, weight of naive) for a player.

        Each table maps a feature bin to (P(dudo), P(calza)).  name=None
        gives the population model.  The weight comes from comparing the
        two hypotheses' predictive log-likelihood on this player so far.
        """
        v = self._call_cache.get(name)
        if v is not None:
            return v
        b0, b1 = self.p["beta_pop"], self.p["beta_call"]
        st = self.stats.get(name) if name is not None else None

        def table(pop_bins, own_bins):
            out = []
            for b in range(CALL_BINS):
                x = (b + 0.5) / CALL_BINS
                pn = pop_bins.n[b]
                pd = (pop_bins.d[b] + b0 * _prior_dudo(x, self.p["call_center"])) / (pn + b0)
                pc = (pop_bins.c[b] + b0 * 0.01) / (pn + b0)
                if own_bins is not None:
                    n = own_bins.n[b]
                    pd = (own_bins.d[b] + b1 * pd) / (n + b1)
                    pc = (own_bins.c[b] + b1 * pc) / (n + b1)
                out.append((pd, pc))
            return out

        naive = table(self.pop.call_naive, st.call_naive if st else None)
        bayes = table(self.pop.call_bayes, st.call_bayes if st else None)
        pop = self.pop
        diff = (pop.ll[0] - pop.ll[1]) / max(1, pop.n_calls) * self.p["w_prior_n"]
        if st is not None:
            diff += st.ll[0] - st.ll[1]
        w = min(0.98, max(0.02, _sigmoid(diff)))
        v = (naive, bayes, w)
        self._call_cache[name] = v
        return v

    def _player_dist(self, name, n, pal, ev, face):
        """Posterior pmf of `name`'s matches for `face` given evidence `ev`."""
        if not ev or n > MAX_DICE:
            return _binom(n, _p6(face, pal))
        key = (name, n, pal, tuple(desc for _, _, desc in ev))
        post = self.post_cache.get(key)
        if post is None:
            w = list(HAND_PRIOR[n])
            for f, lr, _ in ev:
                m = MATCH[n][pal][f]
                for h in range(len(w)):
                    w[h] *= lr[m[h]]
            s = sum(w)
            if s <= 0:
                w = list(HAND_PRIOR[n])
                s = 1.0
            post = {"w": [x / s for x in w]}
            self.post_cache[key] = post
        d = post.get(face)
        if d is None:
            d = [0.0] * (n + 1)
            m = MATCH[n][pal][face]
            w = post["w"]
            for h in range(len(w)):
                d[m[h]] += w[h]
            post[face] = d
        return d

    # -- decision making ---------------------------------------------------

    def play(self, state):
        try:
            action = self._play(state)
            if action in state.legal_actions:
                return action
        except Exception:
            if self.p["strict"]:
                raise
        return _fallback(state)

    def _play(self, state):
        if state.current_bid is None:
            opening = Bid(1, 2)  # Bas always opens with one 2
            if opening in state.legal_actions:
                return opening
        P = self.p
        pal = state.palifico
        players = state.players
        me = players[state.my_id]
        my_name = me.name
        alive = [pl for pl in players if pl.alive and pl.dice_count > 0]
        total = sum(pl.dice_count for pl in alive)
        n_alive = len(alive)
        recovery = all(pl.alive for pl in players)
        my_dice = state.my_dice
        cur = state.current_bid

        # seat order: next alive player after me
        nseat = len(players)
        nxt = None
        for i in range(1, nseat):
            cand = players[(state.my_id + i) % nseat]
            if cand.alive and cand.dice_count > 0:
                nxt = cand
                break

        # previous bidder & evidence from this round's history
        evidence = {}  # name -> list of (face, lr, descriptor)
        prev = None
        bidder = None
        seen = set()
        dice_of = {pl.name: pl.dice_count for pl in players}
        my_ev = {}  # face -> LR of my own first bid on it (how others may read me)
        for rec in state.history:
            a = rec.action
            if isinstance(a, Bid):
                n = dice_of.get(rec.player, 0)
                if rec.player == my_name and a.face not in my_ev and 0 < n <= MAX_DICE:
                    t = _bid_type(a, prev, total, pal, P["use_excess"])
                    my_ev[a.face] = self._likelihood_ratio(my_name, t, _p6(a.face, pal), n)
                if rec.player != my_name and P["use_evidence"] and 0 < n <= MAX_DICE:
                    ev_list = evidence.setdefault(rec.player, [])
                    kf = (rec.player, a.face)
                    if kf not in seen:
                        seen.add(kf)
                        t = _bid_type(a, prev, total, pal, P["use_excess"])
                        p6 = _p6(a.face, pal)
                        lr = self._likelihood_ratio(rec.player, t, p6, n)
                        ev_list.append((a.face, lr, ("bid", a.face, t)))
                    if prev is not None and P["accept_pow"] > 0:
                        # They let `prev` stand instead of calling Dudo.
                        calls = self._call_model(rec.player)[0]
                        p6 = _p6(prev.face, pal)
                        ap = P["accept_pow"]
                        lr = [
                            max(0.02, 1.0 - calls[_call_bin(_binom_tail(total - n, prev.quantity - k, p6))][0]) ** ap
                            for k in range(n + 1)
                        ]
                        ev_list.append((prev.face, lr, ("acc", prev.face, prev.quantity)))
                prev = a
                bidder = rec.player

        opps = [pl for pl in alive if pl.name != my_name]

        # value model
        dv = _ValueModel(alive, my_name, P["gamma"])
        v_lose = dv.delta(my_name, -1)
        v_gain = dv.delta(my_name, +1) if recovery else 0.0

        # per-face distributions of opponents' matching dice
        face_cache = {}

        def opp_dist(pl, face):
            return self._player_dist(pl.name, pl.dice_count, pal, evidence.get(pl.name), face)

        def face_info(face):
            fi = face_cache.get(face)
            if fi is not None:
                return fi
            dists = {pl.name: opp_dist(pl, face) for pl in opps}
            all_d = [1.0]
            rest_d = [1.0]
            for pl in opps:
                all_d = _conv(all_d, dists[pl.name])
                if nxt is None or pl.name != nxt.name:
                    rest_d = _conv(rest_d, dists[pl.name])
            mine = sum(1 for d in my_dice if d == face or (not pal and face != PACO and d == PACO))
            fi = (mine, all_d, rest_d, dists)
            face_cache[face] = fi
            return fi

        def p_true(bid):
            mine, all_d, _, _ = face_info(bid.face)
            return _tail(all_d, bid.quantity - mine)

        def p_exact(bid):
            mine, all_d, _, _ = face_info(bid.face)
            k = bid.quantity - mine
            return all_d[k] if 0 <= k < len(all_d) else 0.0

        options = []  # (ev, tiebreak, action)

        if cur is not None:
            pt = p_true(cur)
            self._last_pt = pt
            v_bidder = dv.delta(bidder, -1) if bidder else 0.0
            ev_dudo = pt * v_lose + (1 - pt) * v_bidder
            options.append((ev_dudo, 0, Dudo()))
            if P["calza_ok"]:
                pe = p_exact(cur)
                ev_calza = pe * v_gain + (1 - pe) * v_lose
                options.append((ev_calza, -1, Calza()))

        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        if bids:
            by_face = {}
            for b in bids:
                by_face.setdefault(b.face, []).append(b)
            avg_opp_lose = sum(dv.delta(pl.name, -1) for pl in opps) / max(1, len(opps))
            v_next_lose = dv.delta(nxt.name, -1) if nxt else 0.0
            v_next_gain = dv.delta(nxt.name, +1) if (nxt and recovery) else 0.0
            if nxt:
                calls_n, calls_b, w_naive = self._call_model(nxt.name)
            n_next = nxt.dice_count if nxt else 0
            n_me = len(my_dice)
            view_cache = {}

            def next_view_tail(face, t, p6, rest_d):
                """Suffix sums of the next player's belief about everyone
                but themselves -- with my hand read through my bids."""
                key = (face, t if face not in my_ev else None)
                v = view_cache.get(key)
                if v is not None:
                    return v
                d = list(_binom(n_me, p6))
                lrs = []
                if face in my_ev:
                    lrs.append(my_ev[face])
                elif n_me <= MAX_DICE:
                    lrs.append(self._likelihood_ratio(my_name, t, p6, n_me))
                for lr in lrs:
                    d = [x * lr[i] for i, x in enumerate(d)]
                z = sum(d)
                if z > 0:
                    d = [x / z for x in d]
                full = _conv(rest_d, d)
                suf = [0.0] * (len(full) + 1)
                for i in range(len(full) - 1, -1, -1):
                    suf[i] = suf[i + 1] + full[i]
                view_cache[key] = suf
                return suf

            for face, fb in by_face.items():
                fb.sort(key=lambda b: b.quantity)
                mine, all_d, rest_d, dists = face_info(face)
                p6 = _p6(face, pal)
                exp_total = mine + sum(i * x for i, x in enumerate(all_d))
                limit = max(fb[0].quantity + P["max_steps"], int(exp_total) + 2)
                next_d = dists[nxt.name] if nxt else [1.0]
                for b in fb:
                    if b.quantity > limit:
                        break
                    q = b.quantity
                    psi = _binom_tail(total, q, p6)  # public plausibility
                    r_back = self._r_back(n_alive, psi)
                    cont = r_back * v_lose + (1 - r_back) * avg_opp_lose
                    if nxt:
                        suf = next_view_tail(face, _bid_type(b, cur, total, pal, P["use_excess"]), p6, rest_d)
                    ev = 0.0
                    for k, pk in enumerate(next_d):
                        if pk <= 0.0:
                            continue
                        # P(bid false | next player holds k)
                        need = q - mine - k
                        p_false = 1.0 - _tail(rest_d, need)
                        p_ex = rest_d[need] if 0 <= need < len(rest_d) else 0.0
                        if nxt:
                            xn = _binom_tail(total - n_next, q - k, p6)
                            j = q - k
                            xb = 1.0 if j <= 0 else (suf[j] if j < len(suf) else 0.0)
                            pdn, pc = calls_n[_call_bin(xn)]
                            pd = w_naive * pdn + (1 - w_naive) * calls_b[_call_bin(xb)][0]
                        else:
                            pd, pc = 0.0, 0.0
                        pc = min(pc, 1.0 - pd)
                        ev_d = p_false * v_lose + (1 - p_false) * v_next_lose
                        ev_c = p_ex * v_next_gain + (1 - p_ex) * v_next_lose
                        ev += pk * (pd * ev_d + pc * ev_c + (1 - pd - pc) * cont)
                    options.append((ev, 1, b))

        options.sort(key=lambda o: (o[0], o[1]), reverse=True)
        self._last_options = options
        choice = options[0][2]
        eps = P["explore_eps"]
        if eps > 0 and isinstance(choice, Bid):
            best = options[0][0]
            near = [o[2] for o in options if isinstance(o[2], Bid) and o[0] >= best - eps]
            choice = self.rng.choice(near)
        if isinstance(choice, Bid):
            psi = _binom_tail(total, choice.quantity, _p6(choice.face, pal))
            self.my_bids.append((_rb_n(n_alive), _rb_bin(psi)))
        return choice


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _count(dice, face, palifico):
    """How many of `dice` count towards a bid on `face` (mirrors rules.py)."""
    if palifico or face == PACO:
        return sum(1 for d in dice if d == face)
    return sum(1 for d in dice if d == face or d == PACO)


def _min_quantity(face, prev, palifico):
    """Smallest legal quantity for `face` on top of `prev` (mirrors rules.py)."""
    if prev is None:
        return 1
    q0, f0 = prev.quantity, prev.face
    if palifico:
        return q0 + 1
    if f0 == PACO:
        return q0 + 1 if face == PACO else 2 * q0 + 1
    if face == PACO:
        return (q0 + 1) // 2
    return q0 if face > f0 else q0 + 1


def _bid_type(bid, prev, total, palifico, use_excess=True):
    """Category of a bid, used as the key for the learned honesty model."""
    if prev is None:
        t = "open"
        p6 = _p6(bid.face, palifico)
        excess = bid.quantity - int(total * p6 / 6.0)
        eb = 0 if excess <= -2 else (1 if excess <= 0 else 2)
    else:
        t = "same" if prev.face == bid.face else "new"
        excess = bid.quantity - _min_quantity(bid.face, prev, palifico)
        eb = 0 if excess <= 0 else (1 if excess == 1 else 2)
    return f"{t}{eb}" if use_excess else t


class _ValueModel:
    """W = d_me^g / sum_j d_j^g -- a crude but smooth P(win) proxy."""

    def __init__(self, alive, my_name, gamma):
        self.d = {pl.name: pl.dice_count for pl in alive}
        self.me = my_name
        self.g = gamma
        self.base = self._w(self.d)

    def _w(self, d):
        mine = d.get(self.me, 0)
        if mine <= 0:
            return 0.0
        tot = sum(v ** self.g for v in d.values() if v > 0)
        return mine ** self.g / tot

    def delta(self, name, change):
        if name not in self.d:
            return 0.0
        d = dict(self.d)
        d[name] = min(MAX_DICE, max(0, d[name] + change))
        return self._w(d) - self.base


def _fallback(state):
    """Dead simple, always-legal policy used only if something went wrong."""
    legal = state.legal_actions
    bid = state.current_bid
    bids = [a for a in legal if isinstance(a, Bid)]
    if bid is not None:
        total = sum(p.dice_count for p in state.players)
        mine = sum(
            1 for d in state.my_dice
            if d == bid.face or (not state.palifico and bid.face != PACO and d == PACO)
        )
        p6 = _p6(bid.face, state.palifico)
        if not bids or _binom_tail(total - state.my_dice_count, bid.quantity - mine, p6) < 0.3:
            for a in legal:
                if isinstance(a, Dudo):
                    return a
    if bids:
        return min(bids, key=lambda b: (b.quantity * (2 if b.face == PACO else 1), b.face))
    return legal[0]
