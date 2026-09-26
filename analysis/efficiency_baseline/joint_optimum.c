/* Exact joint optimum for a matrixgame board.
 *
 * Same problem and same algorithm as joint_optimum.py, roughly 100x faster, so
 * the whole 60-board set finishes in minutes instead of hours.
 *
 * A* over joint configurations.  The heuristic is the sum of per-object BFS
 * distances, i.e. the stored `optimal_moves` evaluated at the current state.
 * It is consistent, so A* is optimal and f is monotone non-decreasing; when the
 * search is cut off, the current f is a proven lower bound on the optimum.
 *
 * Two implementation points make this cheap:
 *   - A move changes one object's distance-to-goal by exactly +-1 at unit cost,
 *     so a successor's f is either f or f+2.  Only two buckets are ever live,
 *     and C* - h0 is necessarily even.
 *   - Up to 64 free cells means a position fits in 6 bits and a whole
 *     configuration in one uint64, with the occupancy set as a second uint64.
 *
 * Input (stdin), all whitespace-separated integers:
 *   N K                      free cells, objects
 *   for each cell: deg, then deg neighbour indices
 *   K start cell indices
 *   K goal cell indices
 *
 * Output (stdout), one line:
 *   h0 <sum-of-BFS>  opt <optimum or -1>  lb <proven lower bound>  expanded <n>  status <exact|cutoff>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define MAXN 64
#define MAXK 8
#define BITS 6
#define POSMASK 63ULL

static int N, K;
static int deg[MAXN], adj[MAXN][4];
static int startp[MAXK], goalp[MAXK];
static int dist[MAXK][MAXN];

static inline int getpos(uint64_t s, int i) { return (int)((s >> (BITS * i)) & POSMASK); }
static inline uint64_t setpos(uint64_t s, int i, int v) {
    return (s & ~(POSMASK << (BITS * i))) | ((uint64_t)v << (BITS * i));
}

/* ---- closed set: open addressing, linear probing, key 0 means empty ------ */
static uint64_t *tbl;
static size_t tmask;
static size_t tcount;

static int closed_insert(uint64_t state) {          /* 1 if new, 0 if present */
    uint64_t key = state + 1;
    size_t h = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> 32) & tmask;
    while (tbl[h]) {
        if (tbl[h] == key) return 0;
        h = (h + 1) & tmask;
    }
    tbl[h] = key;
    tcount++;
    return 1;
}

/* ---- growable stack of states ------------------------------------------- */
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
    long cap = (argc > 1) ? atol(argv[1]) : 60000000L;      /* max expanded states */
    double timelimit = (argc > 2) ? atof(argv[2]) : 900.0;  /* seconds */

    if (scanf("%d %d", &N, &K) != 2) return 1;
    if (N > MAXN || K > MAXK) { fprintf(stderr, "board too large\n"); return 1; }
    for (int i = 0; i < N; i++) {
        scanf("%d", &deg[i]);
        for (int e = 0; e < deg[i]; e++) scanf("%d", &adj[i][e]);
    }
    for (int i = 0; i < K; i++) scanf("%d", &startp[i]);
    for (int i = 0; i < K; i++) scanf("%d", &goalp[i]);

    for (int i = 0; i < K; i++) bfs(goalp[i], dist[i]);

    uint64_t start = 0, goal = 0;
    int h0 = 0;
    for (int i = 0; i < K; i++) {
        start = setpos(start, i, startp[i]);
        goal  = setpos(goal,  i, goalp[i]);
        if (dist[i][startp[i]] < 0) { printf("h0 -1 opt -1 lb -1 expanded 0 status unreachable\n"); return 0; }
        h0 += dist[i][startp[i]];
    }
    if (start == goal) { printf("h0 %d opt 0 lb 0 expanded 0 status exact\n", h0); return 0; }

    size_t slots = 1024;
    while (slots < (size_t)cap * 2) slots <<= 1;
    tbl = calloc(slots, sizeof(uint64_t));
    if (!tbl) { fprintf(stderr, "cannot allocate closed set\n"); return 2; }
    tmask = slots - 1;

    Stack cur = {0}, nxt = {0};
    push(&cur, start);
    int f = h0;
    long expanded = 0;
    clock_t t0 = clock();

    while (cur.n || nxt.n) {
        if (!cur.n) {                       /* f+1 is never populated */
            Stack tmp = cur; cur = nxt; nxt = tmp;
            nxt.n = 0;
            f += 2;
            continue;
        }
        uint64_t s = cur.v[--cur.n];
        if (!closed_insert(s)) continue;
        expanded++;
        if (s == goal) {
            printf("h0 %d opt %d lb %d expanded %ld status exact\n", h0, f, f, expanded);
            return 0;
        }
        if (expanded >= cap ||
            ((expanded & 0xFFFFF) == 0 &&
             (double)(clock() - t0) / CLOCKS_PER_SEC > timelimit)) {
            printf("h0 %d opt -1 lb %d expanded %ld status cutoff\n", h0, f, expanded);
            return 0;
        }
        uint64_t occ = 0;
        for (int i = 0; i < K; i++) occ |= 1ULL << getpos(s, i);
        for (int i = 0; i < K; i++) {
            int u = getpos(s, i);
            const int *di = dist[i];
            for (int e = 0; e < deg[u]; e++) {
                int v = adj[u][e];
                if (occ & (1ULL << v)) continue;
                uint64_t ns = setpos(s, i, v);
                if (di[v] < di[u]) push(&cur, ns); else push(&nxt, ns);
            }
        }
    }
    printf("h0 %d opt -1 lb %d expanded %ld status exhausted\n", h0, f, expanded);
    return 0;
}
