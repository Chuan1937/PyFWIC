
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////                        source injection                        ////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        // Surface acquisition: a 1-D grid over receiver index plus one source
        // slot. The 2-D grid spent 36k work items for ~224 memory addresses.
__kernel void injSrc(__global float *vx,__global float *vz,
                     __global float *taux, __global float *tauz, __global float *tauxz,
                     __global float *seismogram_vxi,__global float *seismogram_vzi,
                     __global float *seismogram_tauxi, __global float *seismogram_tauzi, __global float *seismogram_tauxzi,
                     int dxr,
                     int sourcex, int sourcez,
                     float vsrcx, float vsrcz,
                     float tsrcx, float tsrcz,
                     int nt
                     )


{
  int k = get_global_id(0) ;

  if(k < n_main_rec){
      int i = rec_surface_const;
      int j = k * dxr;
      seismogram_vxi[nt * Nr + k]  =  vx[i*Nx + (j + rec_surface_var)];
      seismogram_vzi[nt * Nr + k]  =  vz[i*Nx + (j + rec_surface_var)];
      seismogram_tauxi[nt * Nr + k]  =  taux[i*Nx + (j + rec_surface_var)];
      seismogram_tauzi[nt * Nr + k]  =  tauz[i*Nx + (j + rec_surface_var)];
      seismogram_tauxzi[nt * Nr + k]  =  tauxz[i*Nx + (j + rec_surface_var)];
  }

  if (k == n_main_rec){
    int c = sourcez*Nx + sourcex;
    vx[c] += vsrcx;
    vz[c] += vsrcz;
    taux[c] += tsrcx;
    tauz[c] += tsrcz;
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
  int ofs = ts * Nr;

  if (k < n_main_rec){
      Avx[i*Nx + (j+rec_surface_var)] += res_vx[ofs + k];
      Avz[i*Nx + (j+rec_surface_var)] += res_vz[ofs + k];
      Ataux[i*Nx + (j+rec_surface_var)] += res_taux[ofs + k];
      Atauz[i*Nx + (j+rec_surface_var)] += res_tauz[ofs + k];
      Atauxz[i*Nx + (j+rec_surface_var)] += res_tauxz[ofs + k];
  }

}
