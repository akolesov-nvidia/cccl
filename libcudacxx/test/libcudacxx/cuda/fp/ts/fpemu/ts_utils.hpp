#ifndef __TS_UTILS_HPP__
#define __TS_UTILS_HPP__

#if defined __CUDACC__ // CUDA only

    // Macro to catch CUDA errors in CUDA runtime calls
    #define CSC(call)                                                     \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (cudaSuccess != err) {                                         \
            fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err) );        \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

    // Macro to catch CUDA errors in function launches
    #define CHECK_LAUNCH_ERROR()                                          \
    do {                                                                  \
        /* Check synchronous errors, i.e. pre-launch */                   \
        cudaError_t err = cudaGetLastError();                             \
        if (cudaSuccess != err) {                                         \
            fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err) );        \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
        /* Check asynchronous errors, i.e. function failed (ULF) */       \
        err = cudaDeviceSynchronize();                                    \
        if (cudaSuccess != err) {                                         \
            fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString( err) );       \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

    // Class to measure the time of a function
    class Timer 
    {
        public:
        Timer();
        virtual ~Timer();
        void start();
        float stop();
        private:
        cudaEvent_t event_start, event_stop;
    };

    // Constructor for the Timer class
    Timer::Timer() 
    {
        CSC(cudaEventCreate(&event_start));
        CSC(cudaEventCreate(&event_stop));
    }

    // Start the timer
    void Timer::start() 
    {
        CSC(cudaEventRecord(event_start, 0));
    }

    // Stop the timer and return the elapsed time in milliseconds
    float Timer::stop() 
    {
        float time;
        CSC(cudaEventRecord(event_stop, 0));
        CSC(cudaEventSynchronize(event_stop));
        CSC(cudaEventElapsedTime(&time, event_start, event_stop));
        return time;
    }

    // Destructor for the Timer class
    Timer::~Timer() 
    {
        CSC(cudaEventDestroy(event_start));
        CSC(cudaEventDestroy(event_stop));
    }

    #ifndef DEFAULT_DEV
        #define DEFAULT_DEV (0)
    #endif

    #define CUDA_INIT() \
        CSC (cudaFree (0)); \
        int cuda_dev = DEFAULT_DEV; \
        struct cudaDeviceProp cuda_props; \
        CSC (cudaSetDevice (cuda_dev)); \
        CSC (cudaGetDeviceProperties (&cuda_props, cuda_dev));
    #define CUDA_DEVICE_NAME() cuda_props.name

    #define RUN_FUNC(function, dataset_array, len) run_function<function><<<blocksPerGrid, threadsPerBlock>>>(dataset_array, len)
    #define MALLOC(x,t,s) cudaMallocManaged(&x,sizeof(t)*s)
    #define FREE(x) cudaFree(x)

#else // host only

    #include <stddef.h>
    #include <sys/time.h>

    #define CSC(call) 
    #define CHECK_LAUNCH_ERROR()            

    // Function to get the current time in milliseconds
    static __INLINE__ double second (void)
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (double)tv.tv_sec * 1.0e3 + (double)tv.tv_usec * 1.0e-3;
    }

    // Function to get the current CPU clock rate in kHz
    static __INLINE__ double get_clock_rate()
    {
        FILE* fp = fopen("/proc/cpuinfo", "r");
        if (!fp) return 0.0;
        
        char line[256];
        double clock_rate = 0.0;
        
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "cpu MHz", 7) == 0) {
                char* freq = strchr(line, ':');
                if (freq) {
                    clock_rate = atof(freq + 1) * 1000.0; // Convert MHz to kHz
                    break;
                }
            }
        }
        
        fclose(fp);
        return clock_rate;
    }

    #define CUDA_INIT()
    #define CUDA_DEVICE_NAME() "host"
    #define RUN_FUNC(function, dataset_array, len) run_function<function>(dataset_array, len)
    #define MALLOC(x,t,s) x = (t*)malloc(sizeof(t)*s)
    #define FREE(x) free(x)

#endif // __CUDACC__

namespace ts
{
    // Custom implementation for C++17 and earlier
    template<typename T, typename R>
    __HOST_DEVICE_DECL__ T bit_cast(const R value) 
    {
        T dst;
        // memcpy implementation
        std::memcpy(static_cast<void*>(&dst), static_cast<const void*>(&value), sizeof(T));
        return dst;
    }
} // end of namespace ts

#endif // __TS_UTILS_HPP__