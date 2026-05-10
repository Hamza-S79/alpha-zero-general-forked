/*
 * Parity test: replay the fixture from python/dump_othello_fixture.py and
 * assert the bitboard rules match Python step-for-step on legal_mask, terminal,
 * and apply.
 *
 * Build:
 *   cc -std=c11 -O2 -Wall -Wextra -I include src/othello.c tests/test_othello.c -o build/test_othello
 *
 * Run:
 *   build/test_othello tests/fixture_othello.bin
 */

#include "othello.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char* msg, int rec_idx) {
    fprintf(stderr, "test_othello: FAIL at record %d: %s\n", rec_idx, msg);
    exit(1);
}

static int read_exact(FILE* f, void* buf, size_t n) {
    return fread(buf, 1, n, f) == n ? 0 : -1;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s fixture.bin\n", argv[0]);
        return 2;
    }

    FILE* f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    int32_t n, num;
    if (read_exact(f, &n, 4) || read_exact(f, &num, 4)) {
        fprintf(stderr, "short read on header\n");
        return 2;
    }
    if (n != OTHELLO_N) {
        fprintf(stderr, "fixture N=%d but compiled OTHELLO_N=%d\n", n, OTHELLO_N);
        return 2;
    }

    printf("Loaded %d records (N=%d)\n", num, n);

    int8_t  canonical[OTHELLO_NN];
    uint8_t valids   [OTHELLO_ACTION_SIZE];
    int8_t  next_canon[OTHELLO_NN];
    int32_t terminal_py, action_py;

    int n_passes_seen = 0, n_terminal_seen = 0;

    for (int i = 0; i < num; i++) {
        if (read_exact(f, canonical, OTHELLO_NN)
         || read_exact(f, valids, OTHELLO_ACTION_SIZE)
         || read_exact(f, &terminal_py, 4)
         || read_exact(f, &action_py, 4)
         || read_exact(f, next_canon, OTHELLO_NN)) {
            die("short read on record body", i);
        }

        Board b;
        othello_from_int8(&b, canonical);

        /* 1. legal-mask parity */
        uint8_t c_valids[OTHELLO_ACTION_SIZE];
        othello_valid_moves_vec(&b, c_valids);
        if (memcmp(valids, c_valids, OTHELLO_ACTION_SIZE) != 0) {
            othello_debug_print(&b);
            for (int k = 0; k < OTHELLO_ACTION_SIZE; k++) {
                if (valids[k] != c_valids[k]) {
                    fprintf(stderr, "  action %d: py=%d c=%d\n", k, valids[k], c_valids[k]);
                }
            }
            die("valid_moves mismatch", i);
        }

        /* 2. terminal parity */
        int term_c = othello_terminal(&b);
        if (term_c != terminal_py) {
            fprintf(stderr, "  py terminal=%d c terminal=%d\n", terminal_py, term_c);
            die("terminal mismatch", i);
        }

        if (terminal_py != 0) {
            n_terminal_seen++;
            continue; /* no apply on terminal records */
        }

        /* 3. apply parity */
        if (action_py == OTHELLO_PASS_ACTION) n_passes_seen++;
        if (action_py < 0 || action_py > OTHELLO_PASS_ACTION) {
            die("python emitted out-of-range action on non-terminal record", i);
        }
        othello_apply(&b, action_py);

        Board expected;
        othello_from_int8(&expected, next_canon);
        if (b.p0 != expected.p0 || b.p1 != expected.p1) {
            fprintf(stderr, "  expected next:\n");
            othello_debug_print(&expected);
            fprintf(stderr, "  got next:\n");
            othello_debug_print(&b);
            fprintf(stderr, "  expected p0=%016llx p1=%016llx\n",
                    (unsigned long long)expected.p0, (unsigned long long)expected.p1);
            fprintf(stderr, "  got      p0=%016llx p1=%016llx\n",
                    (unsigned long long)b.p0, (unsigned long long)b.p1);
            die("apply produced wrong next_canonical", i);
        }
    }

    fclose(f);
    printf("OK: %d records (passes=%d, terminals=%d)\n",
           num, n_passes_seen, n_terminal_seen);
    return 0;
}
