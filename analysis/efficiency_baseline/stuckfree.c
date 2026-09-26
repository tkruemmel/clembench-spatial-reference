/* Attainable optimum under the n-player protocol: C port of stuckfree.py.
 *
 * Same rules and the same search as stuckfree.solve, with joint_optimum.c's
 * two-bucket A* (a move changes f by 0 or +2; a `done` costs 0 and leaves h
 * unchanged, so it stays in the current bucket).  Within one bucket every
 * copy of a state has the same g, so closing a state on first pop is exact.
 *
 * State, one uint64: 6 bits per object position (K <= 6), then a done bit per
 * player (P <= 6), then 3 bits for the player to move.
 *
 * Input (stdin), whitespace-separated integers:
 *   N K P
 *   for each cell: deg, then deg neighbour indices
 *   K start cell indices, K goal cell indices
 *   for each player in turn order: count, then its object indices
 *
 * Output: opt <moves or -1> expanded <n> status <exact|cutoff|none>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define MAXN 64
#define MAXK 6
#define MAXP 6
#define BITS 6
#define POSMASK 63ULL
#define DSHIFT (BITS * MAXK)
#define TSHIFT (DSHIFT + MAXP)

static int N, K, P;
static int deg[MAXN], adj[MAXN][4];
static int startp[MAXK], goalp[MAXK];
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

static uint64_t *tbl;
static size_t tmask;

static int closed_insert(uint64_t state) {
    uint64_t key = state + 1;
    size_t h = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> 32) & tmask;
    while (tbl[h]) {
        if (tbl[h] == key) return 0;
        h = (h + 1) & tmask;
    }
    tbl[h] = key;
    return 1;
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

static void bfs(int src, int *d) {
    static int q[MAXN];
    int head = 0, tail = 0;
    for (int i = 0; i < N; i++) d[i] = -1;
    d[src] = 0; q[tail++] = src;
    while (head < tail) {
        int u = q[head++];
        for (int e = 0; e < deg[u]; e++) {
            int v = adj[u][e];
            if (d[v] < 0) { d[v] = d[u] + 1; q[tail++] = v; }
        }
    }
}

int main(int argc, char **argv) {
    long cap = (argc > 1) ? atol(argv[1]) : 200000000L;
    double timelimit = (argc > 2) ? atof(argv[2]) : 3600.0;

    if (scanf("%d %d %d", &N, &K, &P) != 3) return 1;
    if (N > MAXN || K > MAXK || P > MAXP) { fprintf(stderr, "board too large\n"); return 1; }
    for (int i = 0; i < N; i++) {
        scanf("%d", &deg[i]);
        for (int e = 0; e < deg[i]; e++) scanf("%d", &adj[i][e]);
    }
    for (int i = 0; i < K; i++) scanf("%d", &startp[i]);
    for (int i = 0; i < K; i++) scanf("%d", &goalp[i]);
    for (int p = 0; p < P; p++) {
        scanf("%d", &gsize[p]);
        for (int j = 0; j < gsize[p]; j++) scanf("%d", &gobj[p][j]);
    }
    for (int i = 0; i < K; i++) bfs(goalp[i], dist[i]);

    uint64_t start = 0, goal = 0;
    int h0 = 0;
    for (int i = 0; i < K; i++) {
        start = setpos(start, i, startp[i]);
        goal = setpos(goal, i, goalp[i]);
        h0 += dist[i][startp[i]];
    }
    if (start == goal) { printf("opt 0 expanded 0 status exact\n"); return 0; }

    size_t slots = 1024;
    while (slots < (size_t)cap * 2) slots <<= 1;
    tbl = calloc(slots, sizeof(uint64_t));
    if (!tbl) { fprintf(stderr, "cannot allocate closed set\n"); return 2; }
    tmask = slots - 1;

    Stack cur = {0}, nx = {0};
    push(&cur, mk(start, 0, 0));
    int f = h0;
    long expanded = 0;
    clock_t t0 = clock();

    while (cur.n || nx.n) {
        if (!cur.n) {
            Stack tmp = cur; cur = nx; nx = tmp;
            nx.n = 0;
            f += 2;
            continue;
        }
        uint64_t s = cur.v[--cur.n];
        if (!closed_insert(s)) continue;
        expanded++;
        uint64_t pos = positions(s);
        if (pos == goal) {                  /* the episode ends the instant all are home */
            printf("opt %d expanded %ld status exact\n", f, expanded);
            return 0;
        }
        if (expanded >= cap ||
            ((expanded & 0xFFFFF) == 0 &&
             (double)(clock() - t0) / CLOCKS_PER_SEC > timelimit)) {
            printf("opt -1 lb %d expanded %ld status cutoff\n", f, expanded);
            return 0;
        }
        int done = getdone(s), turn = getturn(s);
        uint64_t occ = 0;
        for (int i = 0; i < K; i++) occ |= 1ULL << getpos(s, i);

        /* option 1: done, if all own objects are home (free, keeps f) */
        int home = 1;
        for (int j = 0; j < gsize[turn]; j++) {
            int i = gobj[turn][j];
            if (getpos(s, i) != goalp[i]) { home = 0; break; }
        }
        if (home) {
            int d2 = done | (1 << turn), t2 = nxt(turn, d2);
            if (t2 >= 0) push(&cur, mk(pos, d2, t2));
        }
        /* option 2: move one own object one cell.  A player with neither option
         * would be aborted; its state simply has no successors. */
        int t2 = nxt(turn, done);
        for (int j = 0; j < gsize[turn]; j++) {
            int i = gobj[turn][j];
            int u = getpos(s, i);
            for (int e = 0; e < deg[u]; e++) {
                int v = adj[u][e];
                if (occ & (1ULL << v)) continue;
                uint64_t ns = mk(setpos(pos, i, v), done, t2);
                if (dist[i][v] < dist[i][u]) push(&cur, ns); else push(&nx, ns);
            }
        }
    }
    printf("opt -1 expanded %ld status none\n", expanded);
    return 0;
}
