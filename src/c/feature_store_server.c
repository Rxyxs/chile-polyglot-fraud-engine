/* feature_store_server.c -- standalone C "Feature Store" process, serving
 * compute_velocity_metrics() over a Windows named shared-memory channel
 * (CreateFileMapping/MapViewOfFile), instead of the in-process ctypes DLL
 * call bridge.py's CVelocityEngine uses.
 *
 * This is a deliberately different IPC mechanism from the Ruby rules
 * engine's stdin/stdout pipe (see bridge.py's RubyRulesEngine): a shared
 * memory ring with a 3-state handshake (idle / request-ready /
 * response-ready), synchronized by busy-waiting on a `volatile long`, not
 * a syscall-backed primitive (no named event/semaphore, no socket). That
 * is the whole point of measuring it separately in
 * src/python/benchmark_throughput.py -- three real IPC/FFI strategies in
 * this repo now, each with a genuinely different latency/throughput
 * profile, not three ways of writing the same thing.
 *
 * Scope, stated plainly: this channel is single-client (one outstanding
 * request at a time -- a second concurrent client would corrupt the
 * handshake). A production feature store would need either N channels
 * (one per client) or a proper multiplexed transport (gRPC, a ring buffer
 * with per-request sequence numbers); out of scope here, and the README
 * says so rather than implying this is production-multi-tenant-ready.
 */
#include "fraud_core.h"
#include <windows.h>
#include <stdio.h>

#define CHANNEL_NAME "FraudFeatureStoreChannel"

/* Deliberately natural alignment, NOT #pragma pack(1): TransactionContext
 * and VelocityMetrics are already defined (unpacked) in fraud_core.h for
 * the existing ctypes DLL bridge (bridge.py), which relies on ctypes'
 * default struct layout matching MSVC's natural alignment for that ABI to
 * work. Packing *this* struct to 1-byte alignment while its two members
 * keep their natural internal layout doesn't touch their internal padding
 * at all (pack() only affects how a struct's own members are placed
 * relative to each other, not a nested type's pre-existing layout) --
 * it just makes `state`, `request`, and `response` misaligned relative to
 * what a straightforward natural-alignment mirror in Python would expect.
 * First version of this file used pack(1) here and produced garbage
 * on every field once read back from Python -- caught by comparing
 * against CVelocityEngine's output for identical input, not by
 * inspection. Fixed by dropping the pragma and matching natural
 * alignment on both sides instead. */
typedef struct {
    volatile long state; /* 0=idle, 1=request-ready, 2=response-ready, 3=shutdown */
    TransactionContext request;
    VelocityMetrics response;
} FeatureStoreChannel;

int main(int argc, char **argv) {
    long max_requests = -1; /* -1 = run until a shutdown signal (state=3) */
    if (argc > 1) {
        max_requests = atol(argv[1]);
    }

    HANDLE hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)sizeof(FeatureStoreChannel), CHANNEL_NAME);
    if (!hMapFile) {
        fprintf(stderr, "CreateFileMapping failed: %lu\n", GetLastError());
        return 1;
    }

    FeatureStoreChannel *channel = (FeatureStoreChannel *)MapViewOfFile(
        hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(FeatureStoreChannel));
    if (!channel) {
        fprintf(stderr, "MapViewOfFile failed: %lu\n", GetLastError());
        CloseHandle(hMapFile);
        return 1;
    }

    channel->state = 0;
    printf("feature_store_server: channel '%s' ready (%zu bytes, max_requests=%ld)\n",
           CHANNEL_NAME, sizeof(FeatureStoreChannel), max_requests);
    fflush(stdout);

    long served = 0;
    for (;;) {
        while (channel->state != 1) {
            if (channel->state == 3) {
                printf("feature_store_server: shutdown signal received, served %ld requests\n", served);
                fflush(stdout);
                goto cleanup;
            }
            Sleep(0); /* yield the timeslice instead of a pure spin */
        }

        compute_velocity_metrics(&channel->request, &channel->response);
        channel->state = 2;
        served++;

        if (max_requests > 0 && served >= max_requests) {
            printf("feature_store_server: served %ld requests (limit reached)\n", served);
            fflush(stdout);
            break;
        }
    }

cleanup:
    UnmapViewOfFile(channel);
    CloseHandle(hMapFile);
    return 0;
}
