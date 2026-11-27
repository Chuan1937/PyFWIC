
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////                        source injection                        ////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        // Surface acquisition: a 1-D grid over receiver index plus one source
        // slot. All shots are processed in the same launch.
__kernel void injSrc(__global float *vx,__global float *vz,
                     __global float *taux, __global float *tauz, __global float *tauxz,
                     __global float *seismogram_vxi,__global float *seismogram_vzi,
                     __global float *seismogram_tauxi, __global float *seismogram_tauzi, __global float *seismogram_tauxzi,
                     int dxr,
                     __global int *sourcex, __global int *sourcez,
                     float vsrcx, float vsrcz,
                     float tsrcx, float tsrcz,
                     int nt
                     )


{
  int k = get_global_id(0) ;
  int i = rec_surface_const;
  int j = k * dxr;
  int s = get_global_id(1);
  int o = s * Nx * Nz;
  int so = nt * Ns * Nr + s * Nr;

    if(k < n_main_rec){
        seismogram_vxi[so + k]  =  vx[o + i*Nx + (j + rec_surface_var)];
        seismogram_vzi[so + k]  =  vz[o + i*Nx + (j + rec_surface_var)];
        seismogram_tauxi[so + k]  =  taux[o + i*Nx + (j + rec_surface_var)];
        seismogram_tauzi[so + k]  =  tauz[o + i*Nx + (j + rec_surface_var)];
        seismogram_tauxzi[so + k]  =  tauxz[o + i*Nx + (j + rec_surface_var)];
    }

  if (k == n_main_rec){
    int c = sourcez[s]*Nx + sourcex[s];
    vx[o + c] += vsrcx;
    vz[o + c] += vsrcz;
    taux[o + c] += tsrcx;
    tauz[o + c] += tsrcz;
  }

}

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////                    adjoint    source injection                        ////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
__kernel void Adj_injSrc(
                         __global float *Avx, __global float *Avz,
                         __global float *Ataux, __global float *Atauz, __global float *Atauxz,
                         __global float *res_vx, __global float * res_vz,
                         __global float *res_taux, __global float *res_tauz, __global float *res_tauxz,
                         int dxr, int ts
                         )


{
  int k = get_global_id(0) ;
  int i = rec_surface_const;
  int j = k * dxr;
  int s = get_global_id(1);
  int o = s * Nx * Nz;
  int ofs = ts * Ns * Nr + s * Nr;

    if (k < n_main_rec){
        Avx[o + i*Nx + (j+rec_surface_var)] += res_vx[ofs + k];
        Avz[o + i*Nx + (j+rec_surface_var)] += res_vz[ofs + k];
        Ataux[o + i*Nx + (j+rec_surface_var)] += res_taux[ofs + k];
        Atauz[o + i*Nx + (j+rec_surface_var)] += res_tauz[ofs + k];
        Atauxz[o + i*Nx + (j+rec_surface_var)] += res_tauxz[ofs + k];
  }

}
