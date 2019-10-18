
typedef struct
{
    float _real;
    float _imag;
} clFloatComplex;


inline clFloatComplex make_clFloatComplex(float r, float i)
{
    clFloatComplex res;
    res._real = r;
    res._imag = i;
    return res;
}

inline float clCrealf(const clFloatComplex x)
{
    return x._real;
}

inline float clCimagf(const clFloatComplex x)
{
    return x._imag;
}

inline clFloatComplex clCaddf(const clFloatComplex x, const clFloatComplex y)
{
    return make_clFloatComplex (clCrealf(x) + clCrealf(y),
        clCimagf(x) + clCimagf(y));
}

inline clFloatComplex clCmulf(const clFloatComplex x, const clFloatComplex y)
{
    clFloatComplex prod;
    prod = make_clFloatComplex((clCrealf(x) * clCrealf(y)) -
                               (clCimagf(x) * clCimagf(y)),
                               (clCrealf(x) * clCimagf(y)) +
                               (clCimagf(x) * clCrealf(y)));
    return prod;
}



// Perform Gridding (Device Function)
// Each thread handles a different grid point
__kernel void d_gridKernel(__global const clFloatComplex *data,
                           const int support,
		                   __global const clFloatComplex *C,
                           __global const int *cOffset,
		                   __global const int *iu,
                           __global const int *iv,
		                   __global clFloatComplex *grid,
                           const int gSize,
                           const int dind)
{
	// The actual starting grid point
	__local int s_gind;

	// The Convoluton function point from which we offset
	__local int s_cind;

	// A copy of the vis data so all threads can read it from shared
	// memory rather than all reading from device memory.
	__local clFloatComplex l_data;
	if (get_local_id(0) == 0) {
		s_gind = iu[dind] + gSize * iv[dind] - support;
		s_cind = cOffset[dind];
	    l_data = data[dind];
	}
    barrier(CLK_LOCAL_MEM_FENCE);
	// Make a local copy from shared memory

	int gind = s_gind;
	int cind = s_cind;

	// blockIdx.x gives the support location in the v direction
	int sSize = 2 * support + 1;
	gind += gSize * get_global_id(0);
	cind += sSize * get_global_id(0);

	// threadIdx.x gives the support location in the u dirction
	grid[gind+get_local_id(0)] = clCaddf(grid[gind+get_local_id(0)], clCmulf(l_data, C[cind+get_local_id(0)]));
}

/*

// Perform De-Gridding (Device Function)
__kernel void d_degridKernel(__global const clFloatComplex *grid,
                             const int gSize,
                             const int support,
                             __global const clFloatComplex *C,
                             __global const int *cOffset,
                             __global const int *iu,
                             __global const int *iv,
                             __global clFloatComplex *data,
                             const int dind,
		                     const int row)
{
    // Constants
    static const int cg_maxSupport = 256;

	// Private data for each thread. Eventually summed by the
	// master thread (i.e. threadIdx.x == 0). Currently 
	__local clFloatComplex s_data[cg_maxSupport];
	s_data[get_local_id(0)] = make_clFloatComplex(0, 0);

	//const int l_dind = dind + blockIdx.x;
	const int l_dind = dind + get_local_id(1);

    // The actual starting grid point
    __local int s_gind;
    // The Convoluton function point from which we offset
    __local int s_cind;

    if (get_local_id(0) == 0) {
            s_gind = iu[l_dind] + gSize * iv[l_dind] - support;
            s_cind = cOffset[l_dind];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Make a local copy from shared memory
    int gind = s_gind;
    int cind = s_cind;

    // row gives the support location in the v direction
    int sSize = 2 * support + 1;
    gind += gSize * row;
    cind += sSize * row;

	// threadIdx.x gives the support location in the u dirction
	s_data[get_local_id(0)] = clCmulf(grid[gind+get_local_id(0)], C[cind+get_local_id(0)]);

	// Sum all the private data elements and accumulate to the
	// device memory
    barrier(CLK_LOCAL_MEM_FENCE);

	if (get_local_id(0) == 0) {
		clFloatComplex sum = make_clFloatComplex(0, 0);
		clFloatComplex original;
		original = data[l_dind];
		for (int i = 0; i < sSize; ++i) {
			sum = clCaddf(sum, s_data[i]);
		}
		original = clCaddf(original, sum);
		data[l_dind] = original;
	}
}
*/
