// FIXME: PARTICLE MOVERS NEED TO BE OVERALLOCATED IN STRUCTORS TO
// ACCOUNT FOR SPLITTING THE MOVER ARRAY BETWEEN HOST AND PIPELINES

#define UNLIKELY_(_c) __builtin_expect((_c),0)
#define DECLARE_ALIGNED_ARRAY_(type,align,name,count)    \
  type name[(count)] __attribute__ ((aligned (align)))

#define IN_spa

#define HAS_V4_PIPELINE
#define HAS_V8_PIPELINE
#define HAS_V16_PIPELINE

#include "spa_private.h"

#include <utility>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../../util/pipelines/pipelines_exec.h"

//----------------------------------------------------------------------------//
// Reference implementation for an advance_p pipeline function which does not
// make use of explicit calls to vector intrinsic functions.
//----------------------------------------------------------------------------//

std::pair<int, int> 
do_moves( const particle_mover_t *from,
          particle_mover_t *to,
          const int max_nm,
          const int n) {  

    int nm = 0;
    int failed_nm = 0;
    for (int i = 0; i < n; ++i) {
        if (from[i].i == -1)
            continue;
        
        if (nm < max_nm) {
            to[nm++] = from[i];
        } else {
            ++failed_nm;
        }
    }

    return std::pair<int, int>{nm, failed_nm};
}

__device__
int move_p_( particle_t      * ALIGNED(128) p0,
            particle_mover_t * ALIGNED(16)  pm,
            accumulator_t    * ALIGNED(128) a0,
            const int64_t    *              g_neighbor,
            const int64_t                   g_rangel,
            const int64_t                   g_rangeh,
            const float                     qsp ) {
  float s_midx, s_midy, s_midz;
  float s_dispx, s_dispy, s_dispz;
  float s_dir[3];
  float v0, v1, v2, v3, v4, v5, q;
  int axis, face;
  int64_t neighbor;
  float *a;
  particle_t * ALIGNED(32) p = p0 + pm->i;

  q = qsp*p->w;

  for(;;) {
    s_midx = p->dx;
    s_midy = p->dy;
    s_midz = p->dz;

    s_dispx = pm->dispx;
    s_dispy = pm->dispy;
    s_dispz = pm->dispz;

    s_dir[0] = (s_dispx>0.0f) ? 1.0f : -1.0f;
    s_dir[1] = (s_dispy>0.0f) ? 1.0f : -1.0f;
    s_dir[2] = (s_dispz>0.0f) ? 1.0f : -1.0f;

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
    v0 = (s_dispx==0.0f) ? 3.4e38f : (s_dir[0]-s_midx)/s_dispx;
    v1 = (s_dispy==0.0f) ? 3.4e38f : (s_dir[1]-s_midy)/s_dispy;
    v2 = (s_dispz==0.0f) ? 3.4e38f : (s_dir[2]-s_midz)/s_dispz;

    // Determine the fractional length and axis of current streak. The
    // streak ends on either the first face intersected by the
    // particle track or at the end of the particle track.
    //
    //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
    //   axis 3        ... streak ends at end of the particle track
    /**/      v3=2.0f, axis=3;
    if(v0<v3) v3=v0,   axis=0;
    if(v1<v3) v3=v1,   axis=1;
    if(v2<v3) v3=v2,   axis=2;
    v3 *= 0.5f;

    // Compute the midpoint and the normalized displacement of the streak
    s_dispx *= v3;
    s_dispy *= v3;
    s_dispz *= v3;
    s_midx += s_dispx;
    s_midy += s_dispy;
    s_midz += s_dispz;

    // Accumulate the streak.  Note: accumulator values are 4 times
    // the total physical charge that passed through the appropriate
    // current quadrant in a time-step
    v5 = q*s_dispx*s_dispy*s_dispz*(1./3.);
    a = (float *)(a0 + p->i);
#   define accumulate_j(X,Y,Z)                                        \
    v4  = q*s_disp##X;    /* v2 = q ux                            */  \
    v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
    v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
    v1 += v4;             /* v1 = q ux (1+dy)                     */  \
    v4  = 1+s_mid##Z;     /* v4 = 1+dz                            */  \
    v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
    v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
    v4  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
    v0 *= v4;             /* v0 = q ux (1-dy)(1-dz)               */  \
    v1 *= v4;             /* v1 = q ux (1+dy)(1-dz)               */  \
    v0 += v5;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
    v1 -= v5;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
    v2 -= v5;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
    v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \
    a[0] += v0;                                                       \
    a[1] += v1;                                                       \
    a[2] += v2;                                                       \
    a[3] += v3
    accumulate_j(x,y,z); a += 4;
    accumulate_j(y,z,x); a += 4;
    accumulate_j(z,x,y);
#   undef accumulate_j

    // Compute the remaining particle displacment
    pm->dispx -= s_dispx;
    pm->dispy -= s_dispy;
    pm->dispz -= s_dispz;

    // Compute the new particle offset
    p->dx += s_dispx+s_dispx;
    p->dy += s_dispy+s_dispy;
    p->dz += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)

    if( axis==3 ) break;

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.

    v0 = s_dir[axis];
    (&(p->dx))[axis] = v0; // Avoid roundoff fiascos--put the particle
                           // _exactly_ on the boundary.
    face = axis; if( v0>0 ) face += 3;
    neighbor = g_neighbor[ 6*p->i + face ];

    if( UNLIKELY_( neighbor==reflect_particles ) ) {
      // Hit a reflecting boundary condition.  Reflect the particle
      // momentum and remaining displacement and keep moving the
      // particle.
      (&(p->ux    ))[axis] = -(&(p->ux    ))[axis];
      (&(pm->dispx))[axis] = -(&(pm->dispx))[axis];
      continue;
    }

    if( UNLIKELY_( neighbor<g_rangel || neighbor>g_rangeh ) ) {
      // Cannot handle the boundary condition here.  Save the updated
      // particle position, face it hit and update the remaining
      // displacement in the particle mover.
      p->i = 8*p->i + face;
      return 1; // Return "mover still in use"
    }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.

    p->i = neighbor - g_rangel; // Compute local index of neighbor
    /**/                         // Note: neighbor - g->rangel < 2^31 / 6
    (&(p->dx))[axis] = -v0;      // Convert coordinate system
  }

  return 0; // Return "mover not in use"
}

__global__
void cuda_advance_p_pipeline_scalar(int nn,
                                    float cdt_dx,
                                    float cdt_dy,
                                    float cdt_dz,
                                    float qdt_2mc,
                                    float qsp,
                                    int64_t g_rangel,
                                    int64_t g_rangeh,
                                    particle_t *p,
                                    accumulator_t *a0,
                                    particle_mover_t *local_pm_arr,
                                    interpolator_t *f0,
                                    particle_t *p0,
                                    int64_t *g_neighbor
                                    ) {

  const interpolator_t * ALIGNED(16)  f;
  float                * ALIGNED(16)  a;
  DECLARE_ALIGNED_ARRAY_( particle_mover_t, 16, local_pm, 1 );
  const float one = 1.0;
  const float one_third = 1.0/3.0;
  const float two_fifteenths = 2.0/15.0;
  float dx, dy, dz, ux, uy, uz, q;
  float hax, hay, haz, cbx, cby, cbz;
  float v0, v1, v2, v3, v4, v5;
  int   ii;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  p += idx;
  for( int n = idx; n < nn; n += stride, p += stride )
  {
    dx   = p->dx;                             // Load position
    dy   = p->dy;
    dz   = p->dz;
    ii   = p->i;

    f    = f0 + ii;                           // Interpolate E

    hax  = qdt_2mc*(    ( f->ex    + dy*f->dexdy    ) +
                     dz*( f->dexdz + dy*f->d2exdydz ) );

    hay  = qdt_2mc*(    ( f->ey    + dz*f->deydz    ) +
                     dx*( f->deydx + dz*f->d2eydzdx ) );

    haz  = qdt_2mc*(    ( f->ez    + dx*f->dezdx    ) +
                     dy*( f->dezdy + dx*f->d2ezdxdy ) );

    cbx  = f->cbx + dx*f->dcbxdx;             // Interpolate B
    cby  = f->cby + dy*f->dcbydy;
    cbz  = f->cbz + dz*f->dcbzdz;

    ux   = p->ux;                             // Load momentum
    uy   = p->uy;
    uz   = p->uz;
    q    = p->w;

    ux  += hax;                               // Half advance E
    uy  += hay;
    uz  += haz;

    v0   = qdt_2mc / sqrtf( one + ( ux*ux + ( uy*uy + uz*uz ) ) );

                                              // Boris - scalars
    v1   = cbx*cbx + ( cby*cby + cbz*cbz );
    v2   = ( v0*v0 ) * v1;
    v3   = v0 * ( one + v2 * ( one_third + v2 * two_fifteenths ) );
    v4   = v3 / ( one + v1 * ( v3 * v3 ) );
    v4  += v4;

    v0   = ux + v3*( uy*cbz - uz*cby );       // Boris - uprime
    v1   = uy + v3*( uz*cbx - ux*cbz );
    v2   = uz + v3*( ux*cby - uy*cbx );

    ux  += v4*( v1*cbz - v2*cby );            // Boris - rotation
    uy  += v4*( v2*cbx - v0*cbz );
    uz  += v4*( v0*cby - v1*cbx );

    ux  += hax;                               // Half advance E
    uy  += hay;
    uz  += haz;

    p->ux = ux;                               // Store momentum
    p->uy = uy;
    p->uz = uz;

    v0   = one / sqrtf( one + ( ux*ux+ ( uy*uy + uz*uz ) ) );
                                              // Get norm displacement

    ux  *= cdt_dx;
    uy  *= cdt_dy;
    uz  *= cdt_dz;

    ux  *= v0;
    uy  *= v0;
    uz  *= v0;

    v0   = dx + ux;                           // Streak midpoint (inbnds)
    v1   = dy + uy;
    v2   = dz + uz;

    v3   = v0 + ux;                           // New position
    v4   = v1 + uy;
    v5   = v2 + uz;

    // FIXME-KJB: COULD SHORT CIRCUIT ACCUMULATION IN THE CASE WHERE QSP==0!
    if (  v3 <= one &&  v4 <= one &&  v5 <= one &&   // Check if inbnds
         -v3 <= one && -v4 <= one && -v5 <= one )
    {
      // Common case (inbnds).  Note: accumulator values are 4 times
      // the total physical charge that passed through the appropriate
      // current quadrant in a time-step.

      q *= qsp;

      p->dx = v3;                             // Store new position
      p->dy = v4;
      p->dz = v5;

      dx = v0;                                // Streak midpoint
      dy = v1;
      dz = v2;

      v5 = q*ux*uy*uz*one_third;              // Compute correction

      a  = (float *)( a0 + ii );              // Get accumulator

#     define ACCUMULATE_JG(X,Y,Z,offset)                                 \
      v4  = q*u##X;   /* v2 = q ux                            */        \
      v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
      v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
      v1 += v4;       /* v1 = q ux (1+dy)                     */        \
      v4  = one+d##Z; /* v4 = 1+dz                            */        \
      v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
      v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
      v4  = one-d##Z; /* v4 = 1-dz                            */        \
      v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
      v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
      v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
      v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
      v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
      v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */        \
      a[offset+0] += v0;                                                \
      a[offset+1] += v1;                                                \
      a[offset+2] += v2;                                                \
      a[offset+3] += v3

      ACCUMULATE_JG( x, y, z, 0 );
      ACCUMULATE_JG( y, z, x, 4 );
      ACCUMULATE_JG( z, x, y, 8 );

#     undef ACCUMULATE_JG
    }

    else                                        // Unlikely
    {
      local_pm->dispx = ux;
      local_pm->dispy = uy;
      local_pm->dispz = uz;

      local_pm->i     = p - p0;

      if ( move_p_( p0, local_pm, a0, g_neighbor, g_rangel, g_rangeh, qsp ) ) // Unlikely
      {
        local_pm_arr[n] = local_pm[0];
      }
    }
  }
}

void
advance_p_pipeline_scalar( advance_p_pipeline_args_t * args,
                           int pipeline_rank,
                           int n_pipeline )
{
  // static __thread cudaStream_t st = nullptr;
  // if (!st)
  //   cudaStreamCreate(&st);

  // allocate cuda memory
  int deviceId = args->w_rank;
  cudaSetDevice(deviceId);
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, deviceId);
  int num_blk_per_sm = 32;
  int num_thread_per_blk = 64;
  // cudaSetDeviceFlags(cudaDeviceMapHost);

  particle_mover_t     *              local_pm_arr;
  particle_mover_t                    init{0, 0, 0, -1};
  particle_t           * ALIGNED(128) p0;
  accumulator_t        * ALIGNED(128) a0;
  interpolator_t       * ALIGNED(128) f0;
  int64_t              *              g_neighbor;

  particle_t           * ALIGNED(32)  p;
  particle_mover_t     * ALIGNED(16)  pm;

  const float qdt_2mc        = args->qdt_2mc;
  const float cdt_dx         = args->cdt_dx;
  const float cdt_dy         = args->cdt_dy;
  const float cdt_dz         = args->cdt_dz;
  const float qsp            = args->qsp;

  int itmp, n, max_nm;
  // Determine which quads of particles quads this pipeline processes.

  DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, itmp, n );

  cudaMallocHost((void **) &local_pm_arr, sizeof(particle_mover_t) * n);
  // cudaMallocHost((void **) &p, sizeof(particle_t) * n);
  // cudaMallocHost((void **) &a0, sizeof(accumulator_t) * (args->n_pipeline + 1) * args->stride);
  // cudaMallocHost((void **) &f0, sizeof(interpolator_t) * args->g->nv);
	// cudaMallocHost((void **) &p0, sizeof(particle_t) * args->np);
  // cudaMallocHost((void **) &g_neighbor, sizeof(int64_t) * 6 * args->g->nv);

  // particle_t        * ALIGNED(32)   d_p;
  // accumulator_t     * ALIGNED(128)  d_a0;
  // particle_mover_t  *               d_local_pm_arr;
  // interpolator_t    * ALIGNED(128)  d_f0;
  // particle_t        * ALIGNED(128)  d_p0;
  // int64_t           *               d_g_neighbor;

  // Determine which quads of particles quads this pipeline processes.

  DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, itmp, n );
  p = args->p0 + itmp;
  a0 = args->a0;
  std::fill_n(local_pm_arr, n, init);
  f0 = args->f0;
  p0 = args->p0;
  g_neighbor = args->g->neighbor;

  cudaHostRegister(p, sizeof(particle_t) * n, cudaHostRegisterDefault);
  cudaHostRegister(a0, sizeof(accumulator_t) * (args->n_pipeline + 1) * args->stride, cudaHostRegisterDefault);
  cudaHostRegister(f0, sizeof(interpolator_t) * args->g->nv, cudaHostRegisterDefault);
	cudaHostRegister(p0, sizeof(particle_t) * args->np, cudaHostRegisterDefault);
  cudaHostRegister(g_neighbor, sizeof(int64_t) * 6 * args->g->nv, cudaHostRegisterDefault);

  // cudaHostGetDevicePointer((void **) &d_p, (void *) p, 0);
  // cudaHostGetDevicePointer((void **) &d_a0, (void *) a0, 0);
  // cudaHostGetDevicePointer((void **) &d_local_pm_arr, (void *) local_pm_arr, 0);
  // cudaHostGetDevicePointer((void **) &d_f0, (void *) f0, 0);
  // cudaHostGetDevicePointer((void **) &d_p0, (void *) p0, 0);
  // cudaHostGetDevicePointer((void **) &d_g_neighbor, (void *) g_neighbor, 0);
  // cudaMemPrefetchAsync(d_p, sizeof(particle_t) * n, deviceId, st);
  // cudaMemPrefetchAsync(d_a0, sizeof(accumulator_t) * (args->n_pipeline + 1) * args->stride, deviceId,st);
  // cudaMemPrefetchAsync(d_local_pm_arr, sizeof(particle_mover_t) * n, deviceId, st);
  // cudaMemPrefetchAsync(d_f0, sizeof(interpolator_t) * args->g->nv, deviceId, st);
  // cudaMemPrefetchAsync(d_p0, sizeof(particle_t) * args->np, deviceId, st);
  // cudaMemPrefetchAsync(d_g_neighbor, sizeof(int64_t) * 6 * args->g->nv, deviceId, st);

  // Determine which movers are reserved for this pipeline.
  // Movers (16 bytes) should be reserved for pipelines in at least
  // multiples of 8 such that the set of particle movers reserved for
  // a pipeline is 128-byte aligned and a multiple of 128-byte in
  // size.  The host is guaranteed to get enough movers to process its
  // particles with this allocation.

  max_nm = args->max_nm - ( args->np&15 );

  if ( max_nm < 0 ) max_nm = 0;

  DISTRIBUTE( max_nm, 8, pipeline_rank, n_pipeline, itmp, max_nm );

  if ( pipeline_rank == n_pipeline ) max_nm = args->max_nm - itmp;

  pm   = args->pm + itmp;
  itmp = 0;

  // Determine which accumulator array to use
  // The host gets the first accumulator array.

  if ( pipeline_rank != n_pipeline ) {
    a0 += ( 1 + pipeline_rank ) *
          POW2_CEIL( (args->nx+2)*(args->ny+2)*(args->nz+2), 2 );
  }

  // cudaHostGetDevicePointer((void **) &d_p, (void *) p, 0);
	// cudaHostGetDevicePointer((void **) &d_a0, (void *) a0, 0);
	// cudaHostGetDevicePointer((void **) &d_local_pm_arr, (void *) local_pm_arr, 0);
  // cudaHostGetDevicePointer((void **) &d_f0, (void *) f0, 0);
  // cudaHostGetDevicePointer((void **) &d_p0, (void *) p0, 0);
  // cudaHostGetDevicePointer((void **) &d_g_neighbor, (void *) g_neighbor, 0);

  // cudaMalloc((void **)&d_p, sizeof(particle_t) * n);
  // cudaMalloc((void **)&d_a0, sizeof(accumulator_t) * (args->n_pipeline + 1) * args->stride);
  // cudaMalloc((void **)&d_local_pm_arr, sizeof(particle_mover_t) * n);
  // cudaMalloc((void **)&d_f0, sizeof(interpolator_t) * args->g->nv);
  // cudaMalloc((void **)&d_p0, sizeof(particle_t) * args->np);
  // cudaMalloc((void **)&d_g_neighbor, sizeof(int64_t) * 6 * args->g->nv);

  // cudaMemcpy(d_p, p, sizeof(particle_t) * n, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_a0, a0, sizeof(accumulator_t) * (args->n_pipeline + 1) * args->stride, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_local_pm_arr, local_pm_arr, sizeof(particle_mover_t) * n, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_f0, f0, sizeof(interpolator_t) * args->g->nv, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_p0, p0, sizeof(particle_t) * args->np, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_g_neighbor, g_neighbor, sizeof(int64_t) * 6 * args->g->nv, cudaMemcpyHostToDevice);

  cuda_advance_p_pipeline_scalar<<<num_sm*num_blk_per_sm, num_thread_per_blk>>>(n,
                                                                                cdt_dx,
                                                                                cdt_dy,
                                                                                cdt_dz,
                                                                                qdt_2mc,
                                                                                qsp,
                                                                                args->g->rangel,
                                                                                args->g->rangeh,
                                                                                p,
                                                                                a0,
                                                                                local_pm_arr,
                                                                                f0,
                                                                                p0,
                                                                                g_neighbor);
  cudaDeviceSynchronize();

  // cudaMemcpy(p, d_p, sizeof(particle_t) * n, cudaMemcpyDeviceToHost);
  // cudaMemcpy(a0, d_a0, sizeof(accumulator_t) * (args->n_pipeline + 1) * args->stride, cudaMemcpyDeviceToHost);  
  // cudaMemcpy(local_pm_arr, d_local_pm_arr, sizeof(particle_mover_t) * n, cudaMemcpyDeviceToHost);

  // perform logic to get updated pm, nm and itmp from local_pm_arr
  std::pair<int, int> result = do_moves(local_pm_arr, pm, max_nm, n);

  args->seg[pipeline_rank].pm        = pm;
  args->seg[pipeline_rank].max_nm    = max_nm;
  args->seg[pipeline_rank].nm        = std::get<0>(result);
  args->seg[pipeline_rank].n_ignored = std::get<1>(result);

  cudaHostUnregister(p);
  cudaHostUnregister(a0);
  cudaFreeHost(local_pm_arr);
  cudaHostUnregister(f0);
	cudaHostUnregister(p0);
  cudaHostUnregister(g_neighbor);

  // cudaFreeHost(p);
  // cudaFreeHost(a0);
  // cudaFreeHost(local_pm_arr);
  // cudaFreeHost(f0);
	// cudaFreeHost(p0);
  // cudaFreeHost(g_neighbor);

  // cudaFree(d_p);
  // cudaFree(d_a0);
  // cudaFree(d_local_pm_arr);
  // cudaFree(d_f0);
  // cudaFree(d_p0);
  // cudaFree(d_g_neighbor);
}

//----------------------------------------------------------------------------//
// Top level function to select and call the proper advance_p pipeline
// function.
//----------------------------------------------------------------------------//

void
advance_p_pipeline( species_t * RESTRICT sp,
                    accumulator_array_t * RESTRICT aa,
                    const interpolator_array_t * RESTRICT ia,
                    int w_rank )
{
  DECLARE_ALIGNED_ARRAY( advance_p_pipeline_args_t, 128, args, 1 );

  DECLARE_ALIGNED_ARRAY( particle_mover_seg_t, 128, seg, MAX_PIPELINE + 1 );

  int rank;

  if ( !sp || !aa || !ia || sp->g != aa->g || sp->g != ia->g )
  {
    ERROR( ( "Bad args" ) );
  }

  args->p0      = sp->p;
  args->pm      = sp->pm;
  args->a0      = aa->a;
  args->f0      = ia->i;
  args->seg     = seg;
  args->g       = sp->g;

  args->qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
  args->cdt_dx  = sp->g->cvac*sp->g->dt*sp->g->rdx;
  args->cdt_dy  = sp->g->cvac*sp->g->dt*sp->g->rdy;
  args->cdt_dz  = sp->g->cvac*sp->g->dt*sp->g->rdz;
  args->qsp     = sp->q;

  args->np      = sp->np;
  args->max_nm  = sp->max_nm;
  args->nx      = sp->g->nx;
  args->ny      = sp->g->ny;
  args->nz      = sp->g->nz;

  args->n_pipeline = aa->n_pipeline;
  args->stride = aa->stride;
  args->w_rank = w_rank;
  // Have the host processor do the last incomplete bundle if necessary.
  // Note: This is overlapped with the pipelined processing.  As such,
  // it uses an entire accumulator.  Reserving an entire accumulator
  // for the host processor to handle at most 15 particles is wasteful
  // of memory.  It is anticipated that it may be useful at some point
  // in the future have pipelines accumulating currents while the host
  // processor is doing other more substantive work (e.g. accumulating
  // currents from particles received from neighboring nodes).
  // However, it is worth reconsidering this at some point in the
  // future.

  EXEC_PIPELINES( advance_p, args, 0 );

  WAIT_PIPELINES();

  // FIXME: HIDEOUS HACK UNTIL BETTER PARTICLE MOVER SEMANTICS
  // INSTALLED FOR DEALING WITH PIPELINES.  COMPACT THE PARTICLE
  // MOVERS TO ELIMINATE HOLES FROM THE PIPELINING.

  sp->nm = 0;
  for( rank = 0; rank <= N_PIPELINE; rank++ )
  {
    if ( args->seg[rank].n_ignored )
    {
      WARNING( ( "Pipeline %i ran out of storage for %i movers",
                 rank, args->seg[rank].n_ignored ) );
    }

    if ( sp->pm + sp->nm != args->seg[rank].pm )
    {
      MOVE( sp->pm + sp->nm, args->seg[rank].pm, args->seg[rank].nm );
    }

    sp->nm += args->seg[rank].nm;
  }
}
