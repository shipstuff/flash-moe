/*
 * fast_expert_io.c — High-throughput expert weight I/O via preadv + pthreads
 *
 * Phase 1: Core batch_read (no readahead, no prefetch)
 *
 * API:
 *   init(num_workers=4)
 *   register_files(file_dict, nocache=False)
 *   register_packed_files(packed_dir, layout)
 *   batch_read(layer, expert_indices)
 *   shutdown()
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* numpy C API */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/uio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <errno.h>

/* macOS QoS */
#include <sys/qos.h>
#include <pthread/qos.h>

/* ---- Constants ---- */

#define FEIO_PAGE_SIZE       16384   /* macOS ARM64 page size */
#define FEIO_MAX_FILES       64      /* max safetensors shards */
#define FEIO_MAX_LAYERS      64      /* max model layers */
#define FEIO_MAX_COMPONENTS  9       /* per layer: 3 proj x 3 attrs */
#define FEIO_MAX_EXPERTS     512     /* max experts per layer */
#define FEIO_MAX_READS       8192    /* max individual read ops per batch */
#define FEIO_MAX_COALESCED   4096    /* max coalesced read groups */
#define FEIO_COALESCE_GAP    65536   /* 64KB: merge reads closer than this */
#define FEIO_MAX_SHAPE_DIMS  4       /* max dims in expert shape */
#define FEIO_STAGING_DEFAULT (256 * 1024 * 1024)  /* 256 MB default staging */
#define FEIO_MAX_IOV         64      /* max iov entries per preadv call */
#define FEIO_COMP_NAME_LEN   48      /* e.g. "gate_proj.weight" */
#define FEIO_PATH_MAX        1024

/* ---- Align helpers ---- */

#define ALIGN_DOWN(x, a) ((x) & ~((a) - 1))
#define ALIGN_UP(x, a)   (((x) + (a) - 1) & ~((a) - 1))

/* ---- Dtype enum ---- */

typedef enum {
    DTYPE_U32 = 0,
    DTYPE_BF16,
    DTYPE_F16,
    DTYPE_F32,
    DTYPE_I32,
    DTYPE_U8,
    DTYPE_COUNT
} DtypeEnum;

static int __attribute__((used)) dtype_elem_size[] = {
    [DTYPE_U32] = 4,
    [DTYPE_BF16] = 2,
    [DTYPE_F16] = 2,
    [DTYPE_F32] = 4,
    [DTYPE_I32] = 4,
    [DTYPE_U8] = 1,
};

static int dtype_to_npy[] = {
    [DTYPE_U32] = NPY_UINT32,
    [DTYPE_BF16] = NPY_UINT16,   /* caller handles BF16->float */
    [DTYPE_F16] = NPY_FLOAT16,
    [DTYPE_F32] = NPY_FLOAT32,
    [DTYPE_I32] = NPY_INT32,
    [DTYPE_U8] = NPY_UINT8,
};

static DtypeEnum parse_dtype(const char *s) {
    if (strcmp(s, "U32") == 0)  return DTYPE_U32;
    if (strcmp(s, "BF16") == 0) return DTYPE_BF16;
    if (strcmp(s, "F16") == 0)  return DTYPE_F16;
    if (strcmp(s, "F32") == 0)  return DTYPE_F32;
    if (strcmp(s, "I32") == 0)  return DTYPE_I32;
    if (strcmp(s, "U8") == 0)   return DTYPE_U8;
    return DTYPE_U32; /* fallback */
}

/* ---- File Registry ---- */

typedef struct {
    int fd;
    char path[FEIO_PATH_MAX];
    int nocache;
} RegisteredFile;

/* ---- Packed-mode component layout (within one expert block) ---- */

typedef struct {
    char comp_name[FEIO_COMP_NAME_LEN]; /* e.g. "gate_proj.weight" */
    size_t offset;                      /* byte offset within expert block */
    size_t size;                        /* byte size of this component */
    DtypeEnum dtype;
    int shape[FEIO_MAX_SHAPE_DIMS];
    int ndim;
} PackedCompMeta;

/* ---- Packed-mode per-layer file descriptor ---- */

typedef struct {
    int fd;                             /* fd for layer_XX.bin */
} PackedLayerFile;

/* ---- Per-component metadata (one per layer per proj.attr) ---- */

typedef struct {
    int file_idx;               /* index into g_files */
    off_t abs_offset;           /* absolute byte offset of fused tensor data */
    size_t expert_stride;       /* bytes between consecutive experts */
    size_t expert_size;         /* bytes for one expert */
    DtypeEnum dtype;
    int shape[FEIO_MAX_SHAPE_DIMS]; /* per-expert shape (e.g. [4096, 128]) */
    int ndim;                   /* number of dims in shape */
    char comp_name[FEIO_COMP_NAME_LEN]; /* e.g. "gate_proj.weight" */
} ComponentMeta;

/* ---- Per-layer metadata ---- */

typedef struct {
    ComponentMeta comps[FEIO_MAX_COMPONENTS];
    int num_comps;
} LayerMeta;

/* ---- Single read request ---- */

typedef struct {
    int file_idx;
    off_t offset;           /* actual data offset (unaligned) */
    size_t length;          /* actual data length */
    off_t aligned_offset;   /* page-aligned start */
    size_t aligned_length;  /* page-aligned length */
    size_t pad_before;      /* bytes to skip at start of aligned read */
    int expert_idx;
    int comp_idx;           /* index into LayerMeta.comps */
    void *dest_buffer;      /* set during allocation phase */
} ReadRequest;

/* ---- Coalesced read group ---- */

typedef struct {
    int file_idx;
    off_t aligned_offset;
    size_t aligned_length;
    void *buffer;           /* staging buffer pointer */
    ReadRequest *requests;  /* array of constituent requests */
    int num_requests;
    int complete;           /* set to 1 by worker */
    ssize_t bytes_read;     /* actual bytes read by preadv */
    int error;              /* errno if read failed */
} CoalescedRead;

/* ---- Worker thread context ---- */

typedef struct {
    pthread_t thread;
    int worker_id;
    int running;            /* 1 while thread should stay alive */

    /* Work items assigned per batch */
    CoalescedRead **work_items;
    int work_count;

    /* Synchronization */
    pthread_mutex_t work_mutex;
    pthread_cond_t work_cond;
    int has_work;           /* flag: new work assigned */

    /* Completion signaling (shared) */
    pthread_mutex_t *done_mutex;
    pthread_cond_t *done_cond;
    int *global_completed;
} WorkerCtx;

/* ---- Staging buffer pool (bump allocator) ---- */

typedef struct {
    void *base;
    size_t total_size;
    size_t used;
} StagingPool;

/* ---- Module global state ---- */

typedef struct {
    RegisteredFile files[FEIO_MAX_FILES];
    int num_files;

    LayerMeta layers[FEIO_MAX_LAYERS];
    int num_layers;

    /* Filename -> file_idx lookup */
    char file_names[FEIO_MAX_FILES][FEIO_PATH_MAX];

    WorkerCtx *workers;
    int num_workers;

    StagingPool staging;

    /* Completion synchronization */
    pthread_mutex_t done_mutex;
    pthread_cond_t done_cond;
    int global_completed;
    int total_work_items;

    int initialized;
    int files_registered;

    /* Packed mode state */
    int use_packed;
    PackedLayerFile packed_layers[FEIO_MAX_LAYERS];
    int packed_num_layers;
    PackedCompMeta packed_comps[FEIO_MAX_COMPONENTS];
    int packed_num_comps;
    size_t packed_expert_size;          /* bytes per expert block (e.g. 7077888) */
} ModuleState;

static ModuleState g_state = {0};

/* ---- Staging pool helpers ---- */

static int staging_init(StagingPool *pool, size_t size) {
    pool->total_size = size;
    pool->used = 0;
    int rc = posix_memalign(&pool->base, FEIO_PAGE_SIZE, size);
    if (rc != 0) {
        pool->base = NULL;
        return -1;
    }
    return 0;
}

static void staging_reset(StagingPool *pool) {
    pool->used = 0;
}

static void *staging_alloc(StagingPool *pool, size_t size) {
    /* Align allocation to page boundary */
    size_t aligned_size = ALIGN_UP(size, FEIO_PAGE_SIZE);
    if (pool->used + aligned_size > pool->total_size) {
        return NULL;
    }
    void *ptr = (char *)pool->base + pool->used;
    pool->used += aligned_size;
    return ptr;
}

static void staging_free(StagingPool *pool) {
    if (pool->base) {
        free(pool->base);
        pool->base = NULL;
    }
    pool->total_size = 0;
    pool->used = 0;
}

/* ---- Worker thread function ---- */

static void *worker_func(void *arg) {
    WorkerCtx *ctx = (WorkerCtx *)arg;

    /* Set QoS to UTILITY (I/O-bound, keep off performance cores) */
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);

    while (ctx->running) {
        /* Wait for work */
        pthread_mutex_lock(&ctx->work_mutex);
        while (!ctx->has_work && ctx->running) {
            pthread_cond_wait(&ctx->work_cond, &ctx->work_mutex);
        }
        if (!ctx->running) {
            pthread_mutex_unlock(&ctx->work_mutex);
            break;
        }

        /* Grab work items */
        CoalescedRead **items = ctx->work_items;
        int count = ctx->work_count;
        ctx->has_work = 0;
        pthread_mutex_unlock(&ctx->work_mutex);

        /* Execute reads */
        for (int i = 0; i < count; i++) {
            CoalescedRead *cr = items[i];
            int fd = g_state.files[cr->file_idx].fd;

            /*
             * Use preadv to scatter-gather into the staging buffer.
             * For a coalesced group, we read the entire contiguous region
             * into one buffer, then individual requests index into it.
             */
            struct iovec iov;
            iov.iov_base = cr->buffer;
            iov.iov_len = cr->aligned_length;

            ssize_t nread = preadv(fd, &iov, 1, cr->aligned_offset);
            if (nread < 0) {
                cr->error = errno;
                cr->bytes_read = -1;
            } else {
                cr->bytes_read = nread;
                cr->error = 0;
            }
            cr->complete = 1;
        }

        /* Signal completion */
        pthread_mutex_lock(ctx->done_mutex);
        (*ctx->global_completed) += count;
        pthread_cond_signal(ctx->done_cond);
        pthread_mutex_unlock(ctx->done_mutex);
    }

    return NULL;
}

/* ---- init(num_workers=4) ---- */

static PyObject *feio_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"num_workers", "staging_mb", NULL};
    int num_workers = 4;
    int staging_mb = 256;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ii", kwlist,
                                     &num_workers, &staging_mb))
        return NULL;

    if (g_state.initialized) {
        PyErr_SetString(PyExc_RuntimeError, "fast_expert_io already initialized");
        return NULL;
    }

    if (num_workers < 1 || num_workers > 32) {
        PyErr_SetString(PyExc_ValueError, "num_workers must be 1-32");
        return NULL;
    }

    /* Allocate staging buffer */
    size_t staging_size = (size_t)staging_mb * 1024 * 1024;
    if (staging_init(&g_state.staging, staging_size) != 0) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate staging buffer");
        return NULL;
    }

    /* Initialize completion sync */
    pthread_mutex_init(&g_state.done_mutex, NULL);
    pthread_cond_init(&g_state.done_cond, NULL);
    g_state.global_completed = 0;
    g_state.total_work_items = 0;

    /* Create worker threads */
    g_state.num_workers = num_workers;
    g_state.workers = (WorkerCtx *)calloc(num_workers, sizeof(WorkerCtx));
    if (!g_state.workers) {
        staging_free(&g_state.staging);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate worker contexts");
        return NULL;
    }

    for (int i = 0; i < num_workers; i++) {
        WorkerCtx *w = &g_state.workers[i];
        w->worker_id = i;
        w->running = 1;
        w->has_work = 0;
        w->work_items = NULL;
        w->work_count = 0;
        w->done_mutex = &g_state.done_mutex;
        w->done_cond = &g_state.done_cond;
        w->global_completed = &g_state.global_completed;

        pthread_mutex_init(&w->work_mutex, NULL);
        pthread_cond_init(&w->work_cond, NULL);

        int rc = pthread_create(&w->thread, NULL, worker_func, w);
        if (rc != 0) {
            PyErr_Format(PyExc_RuntimeError,
                         "Failed to create worker thread %d: %s", i, strerror(rc));
            /* Clean up already-created threads */
            for (int j = 0; j < i; j++) {
                g_state.workers[j].running = 0;
                pthread_mutex_lock(&g_state.workers[j].work_mutex);
                g_state.workers[j].has_work = 1;  /* wake it up to exit */
                pthread_cond_signal(&g_state.workers[j].work_cond);
                pthread_mutex_unlock(&g_state.workers[j].work_mutex);
                pthread_join(g_state.workers[j].thread, NULL);
                pthread_mutex_destroy(&g_state.workers[j].work_mutex);
                pthread_cond_destroy(&g_state.workers[j].work_cond);
            }
            free(g_state.workers);
            g_state.workers = NULL;
            staging_free(&g_state.staging);
            return NULL;
        }
    }

    g_state.initialized = 1;

    Py_RETURN_NONE;
}

/* ---- register_files(file_dict, nocache=False) ---- */
/*
 * file_dict: {filename: full_path, ...}
 * Also parses per-layer component metadata from expert_index_data.
 */

static PyObject *feio_register_files(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"file_dict", "expert_index_data", "nocache", NULL};
    PyObject *file_dict = NULL;
    PyObject *expert_index_data = NULL;
    int nocache = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p", kwlist,
                                     &file_dict, &expert_index_data, &nocache))
        return NULL;

    if (!g_state.initialized) {
        PyErr_SetString(PyExc_RuntimeError, "Call init() first");
        return NULL;
    }

    if (!PyDict_Check(file_dict)) {
        PyErr_SetString(PyExc_TypeError, "file_dict must be a dict");
        return NULL;
    }

    if (!PyDict_Check(expert_index_data)) {
        PyErr_SetString(PyExc_TypeError, "expert_index_data must be a dict");
        return NULL;
    }

    /* --- Open files --- */
    g_state.num_files = 0;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(file_dict, &pos, &key, &value)) {
        if (g_state.num_files >= FEIO_MAX_FILES) {
            PyErr_SetString(PyExc_OverflowError, "Too many files");
            return NULL;
        }
        const char *filename = PyUnicode_AsUTF8(key);
        const char *filepath = PyUnicode_AsUTF8(value);
        if (!filename || !filepath) return NULL;

        int fd = open(filepath, O_RDONLY);
        if (fd < 0) {
            PyErr_Format(PyExc_OSError, "Cannot open %s: %s",
                         filepath, strerror(errno));
            return NULL;
        }

        if (nocache) {
            fcntl(fd, F_NOCACHE, 1);
        }

        int idx = g_state.num_files;
        g_state.files[idx].fd = fd;
        g_state.files[idx].nocache = nocache;
        strncpy(g_state.files[idx].path, filepath, FEIO_PATH_MAX - 1);
        g_state.files[idx].path[FEIO_PATH_MAX - 1] = '\0';
        strncpy(g_state.file_names[idx], filename, FEIO_PATH_MAX - 1);
        g_state.file_names[idx][FEIO_PATH_MAX - 1] = '\0';
        g_state.num_files++;
    }

    /* --- Parse expert_reads from expert_index_data --- */
    PyObject *expert_reads = PyDict_GetItemString(expert_index_data, "expert_reads");
    if (!expert_reads || !PyDict_Check(expert_reads)) {
        PyErr_SetString(PyExc_ValueError,
                        "expert_index_data must contain 'expert_reads' dict");
        return NULL;
    }

    g_state.num_layers = 0;
    pos = 0;
    PyObject *layer_key, *layer_val;
    while (PyDict_Next(expert_reads, &pos, &layer_key, &layer_val)) {
        long layer_idx;
        if (PyLong_Check(layer_key)) {
            layer_idx = PyLong_AsLong(layer_key);
        } else {
            const char *s = PyUnicode_AsUTF8(layer_key);
            if (!s) return NULL;
            layer_idx = atol(s);
        }

        if (layer_idx < 0 || layer_idx >= FEIO_MAX_LAYERS) {
            PyErr_Format(PyExc_ValueError, "Layer index %ld out of range", layer_idx);
            return NULL;
        }

        if (!PyDict_Check(layer_val)) {
            PyErr_Format(PyExc_TypeError, "Layer %ld data must be a dict", layer_idx);
            return NULL;
        }

        LayerMeta *lm = &g_state.layers[layer_idx];
        lm->num_comps = 0;

        PyObject *comp_key, *comp_val;
        Py_ssize_t cpos = 0;
        while (PyDict_Next(layer_val, &cpos, &comp_key, &comp_val)) {
            if (lm->num_comps >= FEIO_MAX_COMPONENTS) {
                PyErr_SetString(PyExc_OverflowError, "Too many components per layer");
                return NULL;
            }

            const char *comp_name = PyUnicode_AsUTF8(comp_key);
            if (!comp_name) return NULL;

            ComponentMeta *cm = &lm->comps[lm->num_comps];
            strncpy(cm->comp_name, comp_name, FEIO_COMP_NAME_LEN - 1);
            cm->comp_name[FEIO_COMP_NAME_LEN - 1] = '\0';

            /* Parse file */
            PyObject *py_file = PyDict_GetItemString(comp_val, "file");
            if (!py_file) {
                PyErr_Format(PyExc_ValueError, "Missing 'file' in %s", comp_name);
                return NULL;
            }
            const char *file_str = PyUnicode_AsUTF8(py_file);
            if (!file_str) return NULL;

            /* Find file index */
            cm->file_idx = -1;
            for (int fi = 0; fi < g_state.num_files; fi++) {
                if (strcmp(g_state.file_names[fi], file_str) == 0) {
                    cm->file_idx = fi;
                    break;
                }
            }
            if (cm->file_idx < 0) {
                PyErr_Format(PyExc_ValueError,
                             "File '%s' not found in registered files", file_str);
                return NULL;
            }

            /* Parse abs_offset */
            PyObject *py_abs = PyDict_GetItemString(comp_val, "abs_offset");
            if (!py_abs) {
                PyErr_Format(PyExc_ValueError, "Missing 'abs_offset' in %s", comp_name);
                return NULL;
            }
            cm->abs_offset = (off_t)PyLong_AsLongLong(py_abs);

            /* Parse expert_stride */
            PyObject *py_stride = PyDict_GetItemString(comp_val, "expert_stride");
            if (!py_stride) {
                PyErr_Format(PyExc_ValueError, "Missing 'expert_stride' in %s", comp_name);
                return NULL;
            }
            cm->expert_stride = (size_t)PyLong_AsUnsignedLongLong(py_stride);

            /* Parse expert_size */
            PyObject *py_size = PyDict_GetItemString(comp_val, "expert_size");
            if (!py_size) {
                PyErr_Format(PyExc_ValueError, "Missing 'expert_size' in %s", comp_name);
                return NULL;
            }
            cm->expert_size = (size_t)PyLong_AsUnsignedLongLong(py_size);

            /* Parse dtype */
            PyObject *py_dtype = PyDict_GetItemString(comp_val, "dtype");
            if (!py_dtype) {
                PyErr_Format(PyExc_ValueError, "Missing 'dtype' in %s", comp_name);
                return NULL;
            }
            const char *dtype_str = PyUnicode_AsUTF8(py_dtype);
            if (!dtype_str) return NULL;
            cm->dtype = parse_dtype(dtype_str);

            /* Parse shape_per_expert */
            PyObject *py_shape = PyDict_GetItemString(comp_val, "shape_per_expert");
            if (!py_shape || !PyList_Check(py_shape)) {
                PyErr_Format(PyExc_ValueError,
                             "Missing or invalid 'shape_per_expert' in %s", comp_name);
                return NULL;
            }
            cm->ndim = (int)PyList_Size(py_shape);
            if (cm->ndim > FEIO_MAX_SHAPE_DIMS) {
                PyErr_Format(PyExc_ValueError,
                             "shape_per_expert has too many dims in %s", comp_name);
                return NULL;
            }
            for (int d = 0; d < cm->ndim; d++) {
                cm->shape[d] = (int)PyLong_AsLong(PyList_GetItem(py_shape, d));
            }

            lm->num_comps++;
        }

        if (layer_idx >= g_state.num_layers)
            g_state.num_layers = (int)(layer_idx + 1);
    }

    g_state.files_registered = 1;

    Py_RETURN_NONE;
}

/* ---- register_packed_files(packed_dir, layout) ---- */
/*
 * packed_dir: string path to directory containing layer_XX.bin files
 * layout: dict with keys: expert_size, num_layers, num_experts, components
 *   components: list of dicts with keys: name, offset, size, dtype, shape
 *
 * Opens one fd per layer file. Sets use_packed=1 so batch_read uses
 * the packed code path (1 read per expert instead of 9).
 */

static PyObject *feio_register_packed_files(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"packed_dir", "layout", NULL};
    const char *packed_dir = NULL;
    PyObject *layout = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO", kwlist,
                                     &packed_dir, &layout))
        return NULL;

    if (!g_state.initialized) {
        PyErr_SetString(PyExc_RuntimeError, "Call init() first");
        return NULL;
    }

    if (!PyDict_Check(layout)) {
        PyErr_SetString(PyExc_TypeError, "layout must be a dict");
        return NULL;
    }

    /* Parse expert_size */
    PyObject *py_expert_size = PyDict_GetItemString(layout, "expert_size");
    if (!py_expert_size) {
        PyErr_SetString(PyExc_ValueError, "layout missing 'expert_size'");
        return NULL;
    }
    g_state.packed_expert_size = (size_t)PyLong_AsUnsignedLongLong(py_expert_size);

    /* Parse num_layers */
    PyObject *py_num_layers = PyDict_GetItemString(layout, "num_layers");
    if (!py_num_layers) {
        PyErr_SetString(PyExc_ValueError, "layout missing 'num_layers'");
        return NULL;
    }
    int num_layers = (int)PyLong_AsLong(py_num_layers);
    if (num_layers <= 0 || num_layers > FEIO_MAX_LAYERS) {
        PyErr_Format(PyExc_ValueError, "num_layers %d out of range (1-%d)",
                     num_layers, FEIO_MAX_LAYERS);
        return NULL;
    }

    /* Parse components array */
    PyObject *py_comps = PyDict_GetItemString(layout, "components");
    if (!py_comps || !PyList_Check(py_comps)) {
        PyErr_SetString(PyExc_ValueError, "layout missing or invalid 'components' list");
        return NULL;
    }

    Py_ssize_t num_comps = PyList_Size(py_comps);
    if (num_comps <= 0 || num_comps > FEIO_MAX_COMPONENTS) {
        PyErr_Format(PyExc_ValueError, "components count %zd out of range (1-%d)",
                     num_comps, FEIO_MAX_COMPONENTS);
        return NULL;
    }

    g_state.packed_num_comps = (int)num_comps;

    for (Py_ssize_t ci = 0; ci < num_comps; ci++) {
        PyObject *comp = PyList_GetItem(py_comps, ci);
        if (!comp || !PyDict_Check(comp)) {
            PyErr_Format(PyExc_TypeError, "components[%zd] must be a dict", ci);
            return NULL;
        }

        PackedCompMeta *pcm = &g_state.packed_comps[ci];

        /* name */
        PyObject *py_name = PyDict_GetItemString(comp, "name");
        if (!py_name) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing 'name'", ci);
            return NULL;
        }
        const char *name = PyUnicode_AsUTF8(py_name);
        if (!name) return NULL;
        strncpy(pcm->comp_name, name, FEIO_COMP_NAME_LEN - 1);
        pcm->comp_name[FEIO_COMP_NAME_LEN - 1] = '\0';

        /* offset */
        PyObject *py_off = PyDict_GetItemString(comp, "offset");
        if (!py_off) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing 'offset'", ci);
            return NULL;
        }
        pcm->offset = (size_t)PyLong_AsUnsignedLongLong(py_off);

        /* size */
        PyObject *py_sz = PyDict_GetItemString(comp, "size");
        if (!py_sz) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing 'size'", ci);
            return NULL;
        }
        pcm->size = (size_t)PyLong_AsUnsignedLongLong(py_sz);

        /* dtype */
        PyObject *py_dt = PyDict_GetItemString(comp, "dtype");
        if (!py_dt) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing 'dtype'", ci);
            return NULL;
        }
        const char *dt_str = PyUnicode_AsUTF8(py_dt);
        if (!dt_str) return NULL;
        pcm->dtype = parse_dtype(dt_str);

        /* shape */
        PyObject *py_shape = PyDict_GetItemString(comp, "shape");
        if (!py_shape || !PyList_Check(py_shape)) {
            PyErr_Format(PyExc_ValueError, "components[%zd] missing or invalid 'shape'", ci);
            return NULL;
        }
        pcm->ndim = (int)PyList_Size(py_shape);
        if (pcm->ndim > FEIO_MAX_SHAPE_DIMS) {
            PyErr_Format(PyExc_ValueError, "components[%zd] shape has too many dims", ci);
            return NULL;
        }
        for (int d = 0; d < pcm->ndim; d++) {
            pcm->shape[d] = (int)PyLong_AsLong(PyList_GetItem(py_shape, d));
        }
    }

    /* Open one fd per layer file */
    g_state.packed_num_layers = num_layers;
    for (int li = 0; li < num_layers; li++) {
        char path[FEIO_PATH_MAX];
        snprintf(path, FEIO_PATH_MAX, "%s/layer_%02d.bin", packed_dir, li);

        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            PyErr_Format(PyExc_OSError, "Cannot open %s: %s", path, strerror(errno));
            /* Close already-opened fds */
            for (int k = 0; k < li; k++) {
                close(g_state.packed_layers[k].fd);
                g_state.packed_layers[k].fd = -1;
            }
            g_state.packed_num_layers = 0;
            return NULL;
        }
        g_state.packed_layers[li].fd = fd;
    }

    /* Also set num_layers for batch_read bounds checking */
    g_state.num_layers = num_layers;
    g_state.use_packed = 1;
    g_state.files_registered = 1;

    Py_RETURN_NONE;
}

/* ---- Comparison function for sorting ReadRequests by (file_idx, offset) ---- */

static int cmp_read_requests(const void *a, const void *b) {
    const ReadRequest *ra = (const ReadRequest *)a;
    const ReadRequest *rb = (const ReadRequest *)b;
    if (ra->file_idx != rb->file_idx)
        return ra->file_idx - rb->file_idx;
    if (ra->offset < rb->offset) return -1;
    if (ra->offset > rb->offset) return 1;
    return 0;
}

/* ---- batch_read(layer, expert_indices, expert_index_data) ---- */

static PyObject *feio_batch_read(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"layer", "expert_indices", NULL};
    int layer;
    PyObject *py_expert_indices = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO", kwlist,
                                     &layer, &py_expert_indices))
        return NULL;

    if (!g_state.initialized || !g_state.files_registered) {
        PyErr_SetString(PyExc_RuntimeError, "Call init() and register_files() first");
        return NULL;
    }

    if (layer < 0 || layer >= g_state.num_layers) {
        PyErr_Format(PyExc_ValueError, "Layer %d out of range (0-%d)",
                     layer, g_state.num_layers - 1);
        return NULL;
    }

    if (!PyList_Check(py_expert_indices) && !PyTuple_Check(py_expert_indices)) {
        PyErr_SetString(PyExc_TypeError, "expert_indices must be a list or tuple");
        return NULL;
    }

    Py_ssize_t num_experts = PySequence_Size(py_expert_indices);
    if (num_experts == 0) {
        return PyDict_New();
    }

    /* ======== PACKED MODE: 1 read per expert ======== */
    if (g_state.use_packed) {
        size_t expert_size = g_state.packed_expert_size;
        int fd = g_state.packed_layers[layer].fd;

        /* Parse expert indices into C array */
        int *expert_ids = (int *)malloc(num_experts * sizeof(int));
        if (!expert_ids) { PyErr_NoMemory(); return NULL; }

        for (Py_ssize_t ei = 0; ei < num_experts; ei++) {
            PyObject *py_eidx = PySequence_GetItem(py_expert_indices, ei);
            if (!py_eidx) { free(expert_ids); return NULL; }
            expert_ids[ei] = (int)PyLong_AsLong(py_eidx);
            Py_DECREF(py_eidx);
            if (expert_ids[ei] < 0 || expert_ids[ei] >= FEIO_MAX_EXPERTS) {
                free(expert_ids);
                PyErr_Format(PyExc_ValueError, "Expert index %d out of range", expert_ids[ei]);
                return NULL;
            }
        }

        /* Sort expert indices to enable coalescing of adjacent experts */
        /* We need to track original order for result dict, so sort a copy */
        int *sorted_ids = (int *)malloc(num_experts * sizeof(int));
        if (!sorted_ids) { free(expert_ids); PyErr_NoMemory(); return NULL; }
        memcpy(sorted_ids, expert_ids, num_experts * sizeof(int));
        /* Simple insertion sort (num_experts is small, typically 4-8) */
        for (Py_ssize_t i = 1; i < num_experts; i++) {
            int key = sorted_ids[i];
            Py_ssize_t j = i - 1;
            while (j >= 0 && sorted_ids[j] > key) {
                sorted_ids[j + 1] = sorted_ids[j];
                j--;
            }
            sorted_ids[j + 1] = key;
        }

        /* Build coalesced reads: merge consecutive experts into single reads */
        /* Max groups = num_experts (worst case: no adjacency) */
        typedef struct {
            off_t offset;           /* aligned file offset */
            size_t length;          /* aligned read length */
            int first_expert;       /* first expert index in this run */
            int count;              /* number of consecutive experts */
        } PackedGroup;

        PackedGroup *groups = (PackedGroup *)malloc(num_experts * sizeof(PackedGroup));
        if (!groups) { free(expert_ids); free(sorted_ids); PyErr_NoMemory(); return NULL; }

        int num_groups = 0;
        for (Py_ssize_t i = 0; i < num_experts; i++) {
            int eidx = sorted_ids[i];
            off_t raw_offset = (off_t)eidx * (off_t)expert_size;

            if (num_groups > 0) {
                PackedGroup *prev = &groups[num_groups - 1];
                int prev_last = prev->first_expert + prev->count;
                if (eidx == prev_last) {
                    /* Adjacent: extend previous group */
                    prev->count++;
                    off_t raw_end = raw_offset + (off_t)expert_size;
                    off_t aligned_end = (off_t)ALIGN_UP(raw_end, FEIO_PAGE_SIZE);
                    if (aligned_end > prev->offset + (off_t)prev->length) {
                        prev->length = (size_t)(aligned_end - prev->offset);
                    }
                    continue;
                }
            }

            /* New group */
            off_t aligned_start = (off_t)ALIGN_DOWN(raw_offset, FEIO_PAGE_SIZE);
            off_t aligned_end = (off_t)ALIGN_UP(raw_offset + (off_t)expert_size, FEIO_PAGE_SIZE);
            groups[num_groups].offset = aligned_start;
            groups[num_groups].length = (size_t)(aligned_end - aligned_start);
            groups[num_groups].first_expert = eidx;
            groups[num_groups].count = 1;
            num_groups++;
        }

        /* Allocate staging and build CoalescedRead array for worker dispatch */
        staging_reset(&g_state.staging);

        CoalescedRead *coalesced = (CoalescedRead *)calloc(num_groups, sizeof(CoalescedRead));
        if (!coalesced) {
            free(expert_ids); free(sorted_ids); free(groups);
            PyErr_NoMemory(); return NULL;
        }

        int staging_overflow = 0;
        for (int gi = 0; gi < num_groups; gi++) {
            CoalescedRead *cr = &coalesced[gi];
            cr->file_idx = layer;  /* reuse file_idx field to carry layer index */
            cr->aligned_offset = groups[gi].offset;
            cr->aligned_length = groups[gi].length;
            cr->num_requests = groups[gi].count;
            cr->complete = 0;
            cr->bytes_read = 0;
            cr->error = 0;

            cr->buffer = staging_alloc(&g_state.staging, cr->aligned_length);
            if (!cr->buffer) {
                cr->buffer = NULL;
                int rc = posix_memalign(&cr->buffer, FEIO_PAGE_SIZE, cr->aligned_length);
                if (rc != 0 || !cr->buffer) {
                    for (int k = 0; k < gi; k++) {
                        char *base = (char *)g_state.staging.base;
                        char *p = (char *)coalesced[k].buffer;
                        if (p < base || p >= base + (ptrdiff_t)g_state.staging.total_size) {
                            free(coalesced[k].buffer);
                        }
                    }
                    free(expert_ids); free(sorted_ids); free(groups); free(coalesced);
                    PyErr_NoMemory(); return NULL;
                }
                staging_overflow = 1;
            }
        }

        /* Distribute groups to workers (round-robin) */
        CoalescedRead ***worker_items = (CoalescedRead ***)calloc(
            g_state.num_workers, sizeof(CoalescedRead **));
        int *worker_counts = (int *)calloc(g_state.num_workers, sizeof(int));
        if (!worker_items || !worker_counts) {
            free(expert_ids); free(sorted_ids); free(groups); free(coalesced);
            free(worker_items); free(worker_counts);
            PyErr_NoMemory(); return NULL;
        }

        for (int i = 0; i < g_state.num_workers; i++) {
            worker_items[i] = (CoalescedRead **)calloc(num_groups, sizeof(CoalescedRead *));
            if (!worker_items[i]) {
                for (int j = 0; j < i; j++) free(worker_items[j]);
                free(worker_items); free(worker_counts);
                free(expert_ids); free(sorted_ids); free(groups); free(coalesced);
                PyErr_NoMemory(); return NULL;
            }
        }
        for (int gi = 0; gi < num_groups; gi++) {
            int wid = gi % g_state.num_workers;
            worker_items[wid][worker_counts[wid]++] = &coalesced[gi];
        }

        /* Worker dispatch: we override file_idx -> fd lookup.
         * Workers use g_state.files[cr->file_idx].fd, but for packed mode
         * we stored the layer index in file_idx. We need to temporarily
         * put the packed fd into the files array — OR — we just do the
         * preadv inline here since packed reads are simple.
         *
         * Better approach: store the fd directly. We'll use a small hack:
         * stash the fd in the CoalescedRead via a reinterpret of file_idx.
         * Actually, workers read g_state.files[cr->file_idx].fd.
         * Cleanest: register packed fds into g_state.files temporarily.
         * But that's fragile. Instead, let's just do the reads in a custom
         * dispatch that doesn't go through the worker's file_idx lookup.
         *
         * Simplest correct approach: put the layer fd into files[0] for each
         * group... No, groups share the same layer. Let's just register
         * the packed layer fd as a single file entry and use file_idx=0.
         */

        /* All groups in one batch_read share the same layer fd.
         * Temporarily register it so workers can find it. */
        int saved_fd = -1;
        int saved_nocache = 0;
        if (g_state.num_files > 0) {
            saved_fd = g_state.files[0].fd;
            saved_nocache = g_state.files[0].nocache;
        }
        g_state.files[0].fd = fd;
        g_state.files[0].nocache = 0;
        for (int gi = 0; gi < num_groups; gi++) {
            coalesced[gi].file_idx = 0;
        }

        /* Dispatch */
        pthread_mutex_lock(&g_state.done_mutex);
        g_state.global_completed = 0;
        g_state.total_work_items = num_groups;
        pthread_mutex_unlock(&g_state.done_mutex);

        for (int i = 0; i < g_state.num_workers; i++) {
            WorkerCtx *w = &g_state.workers[i];
            pthread_mutex_lock(&w->work_mutex);
            w->work_items = worker_items[i];
            w->work_count = worker_counts[i];
            w->has_work = (worker_counts[i] > 0) ? 1 : 0;
            if (w->has_work)
                pthread_cond_signal(&w->work_cond);
            pthread_mutex_unlock(&w->work_mutex);
        }

        /* Release GIL and wait */
        Py_BEGIN_ALLOW_THREADS
        pthread_mutex_lock(&g_state.done_mutex);
        while (g_state.global_completed < g_state.total_work_items) {
            pthread_cond_wait(&g_state.done_cond, &g_state.done_mutex);
        }
        pthread_mutex_unlock(&g_state.done_mutex);
        Py_END_ALLOW_THREADS

        /* Restore files[0] */
        if (saved_fd >= 0) {
            g_state.files[0].fd = saved_fd;
            g_state.files[0].nocache = saved_nocache;
        }

        /* Unpack: for each group, split expert blocks into component numpy arrays */
        PyObject *result = PyDict_New();
        if (!result) goto packed_cleanup;

        for (int gi = 0; gi < num_groups; gi++) {
            CoalescedRead *cr = &coalesced[gi];

            if (cr->error != 0) {
                PyErr_Format(PyExc_IOError,
                             "preadv failed on layer %d at offset %lld: %s",
                             layer, (long long)cr->aligned_offset, strerror(cr->error));
                Py_DECREF(result); result = NULL;
                goto packed_cleanup;
            }

            /* Each group has count consecutive experts starting at first_expert */
            for (int ei = 0; ei < groups[gi].count; ei++) {
                int eidx = groups[gi].first_expert + ei;

                /* Expert block starts at raw offset within the file */
                off_t raw_expert_offset = (off_t)eidx * (off_t)expert_size;
                /* Position within the coalesced buffer */
                size_t buf_offset = (size_t)(raw_expert_offset - cr->aligned_offset);

                PyObject *py_eidx_key = PyLong_FromLong(eidx);
                PyObject *inner = PyDict_New();
                if (!inner) {
                    Py_DECREF(py_eidx_key);
                    Py_DECREF(result); result = NULL;
                    goto packed_cleanup;
                }

                for (int ci = 0; ci < g_state.packed_num_comps; ci++) {
                    PackedCompMeta *pcm = &g_state.packed_comps[ci];

                    npy_intp dims[FEIO_MAX_SHAPE_DIMS];
                    for (int d = 0; d < pcm->ndim; d++) {
                        dims[d] = pcm->shape[d];
                    }

                    int npy_type = dtype_to_npy[pcm->dtype];
                    PyObject *arr = PyArray_SimpleNew(pcm->ndim, dims, npy_type);
                    if (!arr) {
                        Py_DECREF(py_eidx_key);
                        Py_DECREF(inner);
                        Py_DECREF(result); result = NULL;
                        goto packed_cleanup;
                    }

                    void *src = (char *)cr->buffer + buf_offset + pcm->offset;
                    void *dst = PyArray_DATA((PyArrayObject *)arr);
                    memcpy(dst, src, pcm->size);

                    PyObject *py_comp_key = PyUnicode_FromString(pcm->comp_name);
                    PyDict_SetItem(inner, py_comp_key, arr);
                    Py_DECREF(py_comp_key);
                    Py_DECREF(arr);
                }

                PyDict_SetItem(result, py_eidx_key, inner);
                Py_DECREF(py_eidx_key);
                Py_DECREF(inner);
            }
        }

packed_cleanup:
        /* Free overflow mallocs */
        {
            char *base = (char *)g_state.staging.base;
            size_t total = g_state.staging.total_size;
            for (int gi = 0; gi < num_groups; gi++) {
                char *p = (char *)coalesced[gi].buffer;
                if (p < base || p >= base + (ptrdiff_t)total) {
                    free(coalesced[gi].buffer);
                }
            }
        }
        for (int i = 0; i < g_state.num_workers; i++) {
            free(worker_items[i]);
        }
        free(worker_items);
        free(worker_counts);
        free(expert_ids);
        free(sorted_ids);
        free(groups);
        free(coalesced);
        (void)staging_overflow;

        return result;
    }
    /* ======== END PACKED MODE ======== */

    LayerMeta *lm = &g_state.layers[layer];
    if (lm->num_comps == 0) {
        PyErr_Format(PyExc_ValueError, "Layer %d has no registered components", layer);
        return NULL;
    }

    /* ---- Phase 1: Build read requests ---- */

    int total_reads = (int)(num_experts * lm->num_comps);
    if (total_reads > FEIO_MAX_READS) {
        PyErr_Format(PyExc_OverflowError,
                     "Too many reads: %d (max %d)", total_reads, FEIO_MAX_READS);
        return NULL;
    }

    ReadRequest *reads = (ReadRequest *)malloc(total_reads * sizeof(ReadRequest));
    if (!reads) {
        PyErr_NoMemory();
        return NULL;
    }

    int read_idx = 0;
    for (Py_ssize_t ei = 0; ei < num_experts; ei++) {
        PyObject *py_eidx = PySequence_GetItem(py_expert_indices, ei);
        if (!py_eidx) {
            free(reads);
            return NULL;
        }
        int expert_idx = (int)PyLong_AsLong(py_eidx);
        Py_DECREF(py_eidx);

        if (expert_idx < 0 || expert_idx >= FEIO_MAX_EXPERTS) {
            free(reads);
            PyErr_Format(PyExc_ValueError, "Expert index %d out of range", expert_idx);
            return NULL;
        }

        for (int ci = 0; ci < lm->num_comps; ci++) {
            ComponentMeta *cm = &lm->comps[ci];
            ReadRequest *r = &reads[read_idx++];

            r->file_idx = cm->file_idx;
            r->offset = cm->abs_offset + (off_t)expert_idx * (off_t)cm->expert_stride;
            r->length = cm->expert_size;
            r->aligned_offset = (off_t)ALIGN_DOWN(r->offset, FEIO_PAGE_SIZE);
            r->pad_before = (size_t)(r->offset - r->aligned_offset);
            r->aligned_length = ALIGN_UP(r->pad_before + r->length, FEIO_PAGE_SIZE);
            r->expert_idx = expert_idx;
            r->comp_idx = ci;
            r->dest_buffer = NULL;
        }
    }

    /* ---- Phase 2: Sort by (file_idx, offset) ---- */
    qsort(reads, read_idx, sizeof(ReadRequest), cmp_read_requests);

    /* ---- Phase 3: Coalesce adjacent reads ---- */

    CoalescedRead *coalesced = (CoalescedRead *)calloc(FEIO_MAX_COALESCED,
                                                        sizeof(CoalescedRead));
    if (!coalesced) {
        free(reads);
        PyErr_NoMemory();
        return NULL;
    }

    /* Temporary: store request indices per coalesced group */
    int *cr_request_indices = (int *)malloc(read_idx * sizeof(int));
    int *cr_group_starts = (int *)malloc((FEIO_MAX_COALESCED + 1) * sizeof(int));
    if (!cr_request_indices || !cr_group_starts) {
        free(reads);
        free(coalesced);
        free(cr_request_indices);
        free(cr_group_starts);
        PyErr_NoMemory();
        return NULL;
    }

    int num_coalesced = 0;
    int req_list_pos = 0;

    for (int i = 0; i < read_idx; i++) {
        ReadRequest *r = &reads[i];

        if (num_coalesced > 0) {
            CoalescedRead *prev = &coalesced[num_coalesced - 1];
            off_t prev_end = prev->aligned_offset + (off_t)prev->aligned_length;
            off_t gap = r->aligned_offset - prev_end;

            if (r->file_idx == prev->file_idx && gap <= (off_t)FEIO_COALESCE_GAP) {
                /* Extend current group */
                off_t new_end = r->aligned_offset + (off_t)r->aligned_length;
                if (new_end > prev->aligned_offset + (off_t)prev->aligned_length) {
                    prev->aligned_length = (size_t)(new_end - prev->aligned_offset);
                }
                prev->num_requests++;
                cr_request_indices[req_list_pos++] = i;
                continue;
            }
        }

        /* Start a new group */
        if (num_coalesced >= FEIO_MAX_COALESCED) {
            free(reads);
            free(coalesced);
            free(cr_request_indices);
            free(cr_group_starts);
            PyErr_SetString(PyExc_OverflowError, "Too many coalesced groups");
            return NULL;
        }
        cr_group_starts[num_coalesced] = req_list_pos;
        CoalescedRead *cr = &coalesced[num_coalesced];
        cr->file_idx = r->file_idx;
        cr->aligned_offset = r->aligned_offset;
        cr->aligned_length = r->aligned_length;
        cr->num_requests = 1;
        cr->complete = 0;
        cr->bytes_read = 0;
        cr->error = 0;
        cr_request_indices[req_list_pos++] = i;
        num_coalesced++;
    }
    cr_group_starts[num_coalesced] = req_list_pos;

    /* ---- Phase 4: Allocate staging buffers ---- */
    staging_reset(&g_state.staging);

    for (int gi = 0; gi < num_coalesced; gi++) {
        CoalescedRead *cr = &coalesced[gi];
        cr->buffer = staging_alloc(&g_state.staging, cr->aligned_length);
        if (!cr->buffer) {
            /* Staging too small -- fall back to per-group malloc */
            cr->buffer = NULL;
            int rc = posix_memalign(&cr->buffer, FEIO_PAGE_SIZE, cr->aligned_length);
            if (rc != 0 || !cr->buffer) {
                /* Cleanup */
                for (int k = 0; k < gi; k++) {
                    /* Only free if it was malloced (outside staging range) */
                    char *base = (char *)g_state.staging.base;
                    char *p = (char *)coalesced[k].buffer;
                    if (p < base || p >= base + (ptrdiff_t)g_state.staging.total_size) {
                        free(coalesced[k].buffer);
                    }
                }
                free(reads);
                free(coalesced);
                free(cr_request_indices);
                free(cr_group_starts);
                PyErr_NoMemory();
                return NULL;
            }
        }
    }

    /* Set up ReadRequest pointers to point into coalesced group arrays */
    for (int gi = 0; gi < num_coalesced; gi++) {
        CoalescedRead *cr = &coalesced[gi];
        int start = cr_group_starts[gi];
        int end = cr_group_starts[gi + 1];
        cr->requests = (ReadRequest **)malloc == NULL ? NULL : &reads[cr_request_indices[start]];
        /* Actually we need a proper array of pointers. Let's just store
           indices and resolve later during unpack. */
        (void)end; /* used below during unpack */
    }

    /* ---- Phase 5: Distribute work to threads (round-robin by file) ---- */

    /* Allocate per-worker work item arrays */
    CoalescedRead ***worker_items = (CoalescedRead ***)calloc(
        g_state.num_workers, sizeof(CoalescedRead **));
    int *worker_counts = (int *)calloc(g_state.num_workers, sizeof(int));
    if (!worker_items || !worker_counts) {
        free(reads);
        free(coalesced);
        free(cr_request_indices);
        free(cr_group_starts);
        free(worker_items);
        free(worker_counts);
        PyErr_NoMemory();
        return NULL;
    }

    for (int i = 0; i < g_state.num_workers; i++) {
        worker_items[i] = (CoalescedRead **)calloc(num_coalesced, sizeof(CoalescedRead *));
        if (!worker_items[i]) {
            for (int j = 0; j < i; j++) free(worker_items[j]);
            free(worker_items);
            free(worker_counts);
            free(reads);
            free(coalesced);
            free(cr_request_indices);
            free(cr_group_starts);
            PyErr_NoMemory();
            return NULL;
        }
    }

    /* Round-robin by coalesced group index -- spread across all workers
     * even when reads are from the same file (common for single-shard layers) */
    for (int gi = 0; gi < num_coalesced; gi++) {
        int wid = gi % g_state.num_workers;
        worker_items[wid][worker_counts[wid]++] = &coalesced[gi];
    }

    /* ---- Phase 6: Dispatch to workers and wait ---- */

    pthread_mutex_lock(&g_state.done_mutex);
    g_state.global_completed = 0;
    g_state.total_work_items = num_coalesced;
    pthread_mutex_unlock(&g_state.done_mutex);

    /* Dispatch */
    for (int i = 0; i < g_state.num_workers; i++) {
        WorkerCtx *w = &g_state.workers[i];
        pthread_mutex_lock(&w->work_mutex);
        w->work_items = worker_items[i];
        w->work_count = worker_counts[i];
        w->has_work = (worker_counts[i] > 0) ? 1 : 0;
        if (w->has_work)
            pthread_cond_signal(&w->work_cond);
        pthread_mutex_unlock(&w->work_mutex);
    }

    /* Release GIL and wait for all workers */
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&g_state.done_mutex);
    while (g_state.global_completed < g_state.total_work_items) {
        pthread_cond_wait(&g_state.done_cond, &g_state.done_mutex);
    }
    pthread_mutex_unlock(&g_state.done_mutex);
    Py_END_ALLOW_THREADS

    /* ---- Phase 7: Unpack results into Python dict ---- */
    /*
     * Result: {expert_idx: {"comp_name": numpy_array, ...}, ...}
     */

    PyObject *result = PyDict_New();
    if (!result) goto cleanup;

    for (int gi = 0; gi < num_coalesced; gi++) {
        CoalescedRead *cr = &coalesced[gi];

        if (cr->error != 0) {
            PyErr_Format(PyExc_IOError,
                         "preadv failed on %s at offset %lld: %s",
                         g_state.files[cr->file_idx].path,
                         (long long)cr->aligned_offset,
                         strerror(cr->error));
            Py_DECREF(result);
            result = NULL;
            goto cleanup;
        }

        int start = cr_group_starts[gi];
        int end = cr_group_starts[gi + 1];

        for (int ri = start; ri < end; ri++) {
            int req_idx = cr_request_indices[ri];
            ReadRequest *r = &reads[req_idx];
            ComponentMeta *cm = &lm->comps[r->comp_idx];

            /* Compute offset within the coalesced buffer */
            size_t buf_offset = (size_t)(r->offset - cr->aligned_offset);

            /* Get or create inner dict for this expert */
            PyObject *py_eidx_key = PyLong_FromLong(r->expert_idx);
            PyObject *inner = PyDict_GetItem(result, py_eidx_key);
            if (!inner) {
                inner = PyDict_New();
                PyDict_SetItem(result, py_eidx_key, inner);
                Py_DECREF(inner); /* SetItem increfs */
            }
            Py_DECREF(py_eidx_key);

            /* Create numpy array from the buffer data */
            npy_intp dims[FEIO_MAX_SHAPE_DIMS];
            for (int d = 0; d < cm->ndim; d++) {
                dims[d] = cm->shape[d];
            }

            int npy_type = dtype_to_npy[cm->dtype];

            /*
             * We need to copy data because the staging buffer is ephemeral.
             * Create a new numpy array and memcpy into it.
             */
            PyObject *arr = PyArray_SimpleNew(cm->ndim, dims, npy_type);
            if (!arr) {
                Py_DECREF(result);
                result = NULL;
                goto cleanup;
            }

            void *src = (char *)cr->buffer + buf_offset;
            void *dst = PyArray_DATA((PyArrayObject *)arr);
            memcpy(dst, src, r->length);

            /* Store in inner dict with comp_name as key */
            PyObject *py_comp_key = PyUnicode_FromString(cm->comp_name);
            PyDict_SetItem(inner, py_comp_key, arr);
            Py_DECREF(py_comp_key);
            Py_DECREF(arr);
        }
    }

cleanup:
    /* Free overflow mallocs (buffers outside staging pool) */
    {
        char *base = (char *)g_state.staging.base;
        size_t total = g_state.staging.total_size;
        for (int gi = 0; gi < num_coalesced; gi++) {
            char *p = (char *)coalesced[gi].buffer;
            if (p < base || p >= base + (ptrdiff_t)total) {
                free(coalesced[gi].buffer);
            }
        }
    }

    for (int i = 0; i < g_state.num_workers; i++) {
        free(worker_items[i]);
    }
    free(worker_items);
    free(worker_counts);
    free(reads);
    free(coalesced);
    free(cr_request_indices);
    free(cr_group_starts);

    return result;
}

/* ---- shutdown() ---- */

static PyObject *feio_shutdown(PyObject *self, PyObject *args) {
    if (!g_state.initialized) {
        Py_RETURN_NONE;
    }

    /* Stop worker threads */
    if (g_state.workers) {
        for (int i = 0; i < g_state.num_workers; i++) {
            WorkerCtx *w = &g_state.workers[i];
            pthread_mutex_lock(&w->work_mutex);
            w->running = 0;
            w->has_work = 1;  /* wake to exit */
            pthread_cond_signal(&w->work_cond);
            pthread_mutex_unlock(&w->work_mutex);
        }
        for (int i = 0; i < g_state.num_workers; i++) {
            pthread_join(g_state.workers[i].thread, NULL);
            pthread_mutex_destroy(&g_state.workers[i].work_mutex);
            pthread_cond_destroy(&g_state.workers[i].work_cond);
        }
        free(g_state.workers);
        g_state.workers = NULL;
    }

    /* Close file descriptors (scattered mode) */
    for (int i = 0; i < g_state.num_files; i++) {
        if (g_state.files[i].fd >= 0) {
            close(g_state.files[i].fd);
            g_state.files[i].fd = -1;
        }
    }
    g_state.num_files = 0;

    /* Close packed layer fds */
    for (int i = 0; i < g_state.packed_num_layers; i++) {
        if (g_state.packed_layers[i].fd >= 0) {
            close(g_state.packed_layers[i].fd);
            g_state.packed_layers[i].fd = -1;
        }
    }
    g_state.packed_num_layers = 0;
    g_state.packed_num_comps = 0;
    g_state.use_packed = 0;

    /* Free staging buffer */
    staging_free(&g_state.staging);

    /* Destroy completion sync */
    pthread_mutex_destroy(&g_state.done_mutex);
    pthread_cond_destroy(&g_state.done_cond);

    g_state.initialized = 0;
    g_state.files_registered = 0;
    g_state.num_layers = 0;

    Py_RETURN_NONE;
}

/* ---- stats() — return diagnostic info ---- */

static PyObject *feio_stats(PyObject *self, PyObject *args) {
    return Py_BuildValue("{s:i, s:i, s:i, s:n, s:n, s:i, s:i, s:i, s:n}",
                         "initialized", g_state.initialized,
                         "num_workers", g_state.num_workers,
                         "num_files", g_state.num_files,
                         "staging_total", (Py_ssize_t)g_state.staging.total_size,
                         "staging_used", (Py_ssize_t)g_state.staging.used,
                         "num_layers", g_state.num_layers,
                         "use_packed", g_state.use_packed,
                         "packed_num_comps", g_state.packed_num_comps,
                         "packed_expert_size", (Py_ssize_t)g_state.packed_expert_size);
}

/* ---- Module definition ---- */

static PyMethodDef feio_methods[] = {
    {"init", (PyCFunction)feio_init, METH_VARARGS | METH_KEYWORDS,
     "init(num_workers=4, staging_mb=256) -- Create worker thread pool"},
    {"register_files", (PyCFunction)feio_register_files, METH_VARARGS | METH_KEYWORDS,
     "register_files(file_dict, expert_index_data, nocache=False) -- Open FDs and parse metadata"},
    {"register_packed_files", (PyCFunction)feio_register_packed_files, METH_VARARGS | METH_KEYWORDS,
     "register_packed_files(packed_dir, layout) -- Open packed layer files and parse layout"},
    {"batch_read", (PyCFunction)feio_batch_read, METH_VARARGS | METH_KEYWORDS,
     "batch_read(layer, expert_indices) -- Read all components for cache-missed experts"},
    {"shutdown", feio_shutdown, METH_NOARGS,
     "shutdown() -- Join workers, close FDs, free buffers"},
    {"stats", feio_stats, METH_NOARGS,
     "stats() -- Return diagnostic info dict"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef feio_module = {
    PyModuleDef_HEAD_INIT,
    "fast_expert_io",
    "High-throughput expert weight I/O via preadv + pthreads (Phase 1)",
    -1,
    feio_methods
};

PyMODINIT_FUNC PyInit_fast_expert_io(void) {
    /* Import numpy C API */
    import_array();

    PyObject *m = PyModule_Create(&feio_module);
    if (!m) return NULL;

    /* Add constants */
    PyModule_AddIntConstant(m, "PAGE_SIZE", FEIO_PAGE_SIZE);
    PyModule_AddIntConstant(m, "COALESCE_GAP", FEIO_COALESCE_GAP);
    PyModule_AddIntConstant(m, "MAX_FILES", FEIO_MAX_FILES);
    PyModule_AddIntConstant(m, "MAX_LAYERS", FEIO_MAX_LAYERS);
    PyModule_AddIntConstant(m, "MAX_EXPERTS", FEIO_MAX_EXPERTS);

    return m;
}
