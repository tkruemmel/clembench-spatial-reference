/* h*(s) from any protocol state: stuckfree.c answering a stream of queries.
 *
 * Same rules and the same two-bucket A* as stuckfree.c, but the search starts
 * from a given state (positions, done set, player to move) instead of the
 * board's start, and one process answers many queries on one board.  The
 * heuristic routes each object around the objects of players that have already
 * declared done: they never move again, so those distances stay a lower bound
 * for the rest of the search, and a target that is cut off is found without
 * searching.  A move still changes the heuristic by exactly one (the grid is
 * bipartite), so the two-bucket invariant of stuckfree.c holds.
 *
 * Input (stdin), whitespace-separated integers:
 *   N K P
 *   for each cell: deg, then deg neighbour indices
 *   K goal cell indices
 *   for each player in turn order: count, then its object indices
 * then any number of queries:
 *   K position cell indices, done mask, player to move, bound (-1 = none)
 *
 * One output line per query:
 *   h <moves> <expanded>     h*(s) = moves <= bound
 *   gt <expanded>            h*(s) > bound
 *   none <expanded>          no solution, h*(s) = infinity
 *   cut <lb> <expanded>      expansion cap reached, h*(s) >= lb
 *
 *   cc -O3 -o analysis/efficiency_baseline/hstar_c analysis/efficiency_baseline/hstar.c
 *   hstar_c [cap]            (cap: expansions per query, default 16000000)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define MAXN 64
#define MAXK 6
#define MAXP 6
#define BITS 6
#define POSMASK 63ULL
#define DSHIFT (BITS * MAXK)
#define TSHIFT (DSHIFT + MAXP)

static int N, K, P;
static int deg[MAXN], adj[MAXN][4];
static int goalp[MAXK];
static int dist[MAXK][MAXN];
static int gsize[MAXP], gobj[MAXP][MAXK];

static inline int getpos(uint64_t s, int i) { return (int)((s >> (BITS * i)) & POSMASK); }
static inline uint64_t setpos(uint64_t s, int i, int v) {
    return (s & ~(POSMASK << (BITS * i))) | ((uint64_t)v << (BITS * i));
}
static inline int getdone(uint64_t s) { return (int)((s >> DSHIFT) & ((1u << MAXP) - 1)); }
static inline int getturn(uint64_t s) { return (int)(s >> TSHIFT); }
static inline uint64_t positions(uint64_t s) { return s & ((1ULL << DSHIFT) - 1); }
static inline uint64_t mk(uint64_t pos, int done, int turn) {
    return pos | ((uint64_t)done << DSHIFT) | ((uint64_t)turn << TSHIFT);
}

static int nxt(int turn, int done) {       /* next player not yet done, -1 if none */
    for (int k = 1; k <= P; k++) {
        int t = (turn + k) % P;
        if (!((done >> t) & 1)) return t;
    }
    return -1;
}

typedef struct { uint64_t *v; size_t n, cap; } Stack;

static void push(Stack *s, uint64_t x) {
    if (s->n == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 1024;
        s->v = realloc(s->v, s->cap * sizeof(uint64_t));
        if (!s->v) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    s->v[s->n++] = x;
}

static uint64_t *tbl;
static size_t tmask;
static Stack touched;                      /* slots to clear before the next query */

static int closed_insert(uint64_t state) {
    uint64_t key = state + 1;
    size_t h = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> 32) & tmask;
    while (tbl[h]) {
        if (tbl[h] == key) return 0;
        h = (h + 1) & tmask;
    }
    tbl[h] = key;
    push(&touched, h);
    return 1;
}

/* BFS from src over cells not in `blocked` (bitmask) */
static void bfs(int src, uint64_t blocked, int *d) {
    static int q[MAXN];
    int head = 0, tail = 0;
    for (int i = 0; i < N; i++) d[i] = -1;
    d[src] = 0; q[tail++] = src;
    while (head < tail) {
        int u = q[head++];
        for (int e = 0; e < deg[u]; e++) {
            int v = adj[u][e];
            if (d[v] < 0 && !((blocked >> v) & 1)) { d[v] = d[u] + 1; q[tail++] = v; }
        }
    }
}

static Stack cur, nx;

static void query(uint64_t pos, int done, int turn, int bound, long cap) {
    uint64_t goal = 0, frozen = 0;
    for (int i = 0; i < K; i++) goal = setpos(goal, i, goalp[i]);
    for (int p = 0; p < P; p++)
        if ((done >> p) & 1)
            for (int j = 0; j < gsize[p]; j++) frozen |= 1ULL << getpos(pos, gobj[p][j]);
    int f = 0;
    for (int i = 0; i < K; i++) {
        bfs(goalp[i], frozen, dist[i]);
        int d = dist[i][getpos(pos, i)];
        if (d < 0) { printf("none 0\n"); return; }
        f += d;
    }
    if (pos == goal) { printf("h 0 0\n"); return; }
    if (bound >= 0 && f > bound) { printf("gt 0\n"); return; }

    cur.n = nx.n = 0;
    push(&cur, mk(pos, done, turn));
    long expanded = 0;
    for (;;) {
        if (!cur.n) {
            if (!nx.n) { printf("none %ld\n", expanded); break; }
            if (bound >= 0 && f + 2 > bound) { printf("gt %ld\n", expanded); break; }
            Stack tmp = cur; cur = nx; nx = tmp;
            nx.n = 0;
            f += 2;
            continue;
        }
        uint64_t s = cur.v[--cur.n];
        if (!closed_insert(s)) continue;
        expanded++;
        uint64_t p = positions(s);
        if (p == goal) { printf("h %d %ld\n", f, expanded); break; }
        if (expanded >= cap) { printf("cut %d %ld\n", f, expanded); break; }
        int dn = getdone(s), t = getturn(s);
        uint64_t occ = 0;
        for (int i = 0; i < K; i++) occ |= 1ULL << getpos(s, i);

        /* option 1: done, if all own objects are home (free, keeps f) */
        int home = 1;
        for (int j = 0; j < gsize[t]; j++) {
            int i = gobj[t][j];
            if (getpos(s, i) != goalp[i]) { home = 0; break; }
        }
        if (home) {
            int d2 = dn | (1 << t), t2 = nxt(t, d2);
            if (t2 >= 0) push(&cur, mk(p, d2, t2));
        }
        /* option 2: move one own object one cell */
        int t2 = nxt(t, dn);
        for (int j = 0; j < gsize[t]; j++) {
            int i = gobj[t][j];
            int u = getpos(s, i);
            for (int e = 0; e < deg[u]; e++) {
                int v = adj[u][e];
                if (occ & (1ULL << v)) continue;
                uint64_t ns = mk(setpos(p, i, v), dn, t2);
                if (dist[i][v] < dist[i][u]) push(&cur, ns); else push(&nx, ns);
            }
        }
    }
    for (size_t k = 0; k < touched.n; k++) tbl[touched.v[k]] = 0;
    touched.n = 0;
}

int main(int argc, char **argv) {
    long cap = (argc > 1) ? atol(argv[1]) : 16000000L;

    if (scanf("%d %d %d", &N, &K, &P) != 3) return 1;
    if (N > MAXN || K > MAXK || P > MAXP) { fprintf(stderr, "board too large\n"); return 1; }
    for (int i = 0; i < N; i++) {
        if (scanf("%d", &deg[i]) != 1) return 1;
        for (int e = 0; e < deg[i]; e++) if (scanf("%d", &adj[i][e]) != 1) return 1;
    }
    for (int i = 0; i < K; i++) if (scanf("%d", &goalp[i]) != 1) return 1;
    for (int p = 0; p < P; p++) {
        if (scanf("%d", &gsize[p]) != 1) return 1;
        for (int j = 0; j < gsize[p]; j++) if (scanf("%d", &gobj[p][j]) != 1) return 1;
    }

    size_t slots = 1024;
    while (slots < (size_t)cap * 2) slots <<= 1;
    tbl = calloc(slots, sizeof(uint64_t));
    if (!tbl) { fprintf(stderr, "cannot allocate closed set\n"); return 2; }
    tmask = slots - 1;

    for (;;) {
        uint64_t pos = 0;
        int done, turn, bound, v;
        for (int i = 0; i < K; i++) {
            if (scanf("%d", &v) != 1) return 0;
            pos = setpos(pos, i, v);
        }
        if (scanf("%d %d %d", &done, &turn, &bound) != 3) return 0;
        query(pos, done, turn, bound, cap);
        fflush(stdout);
    }
}
