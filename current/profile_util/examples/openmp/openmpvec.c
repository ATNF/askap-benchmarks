#include <string.h>
#include <profile_util.h>

#ifdef USEOPENMP
#include <omp.h>
#endif

int foo(int x) {
    if (x<=1) return x;
    else return x+foo(x-1);
}


int main() {
    log_parallel_api();
    log_binding();
    #pragma omp parallel default(none)
    {
        log_thread_affinity();
        #pragma single 
        {
            int value = 0;
            for (auto i=0;i<3; i++) 
            {
                #pragma taskgroup 
                {
                    #pragma task 
                    {
                        value = foo(value+i);
                    }
                }

            }
        }
    }
    log_mem_usage();
}
