__kernel void injSrc(__global float *vx,__global float *vz,
                     __global float *taux, __global float *tauz, __global float *tauxz,
                     __global float *seismogram_vxi,__global float *seismogram_vzi,
                     __global float *seismogram_tauxi, __global float *seismogram_tauzi, __global float *seismogram_tauxzi,
                     int dxr,
                     __global int *sourcex, __global int *sourcez,
                     float vsrcx, float vsrcz,
                     float tsrcx, float tsrcz,
                      int nt)


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;
  int s = get_global_id(2);
  int o = s * Nx * Nz;
  int so = nt * Ns * Nr + s * Nr;

  if (i==sourcez[s] && j==sourcex[s]){
    vx[o + center] += vsrcx;
    vz[o + center] += vsrcz;
    taux[o + center] += tsrcx;
    tauz[o + center] += tsrcz;

  }

   if(i%dxr==0 && j == rec_top_right_const){
    int ir =  i/dxr;
      seismogram_vxi[so + ir]  =  vx[o + (i+rec_top_right_var)*Nx + j];
      seismogram_vzi[so + ir]  =  vx[(i+rec_top_right_var)*Nx + j];
      seismogram_tauxi[so + ir]  =  taux[o + (i+rec_top_right_var)*Nx + j];
      seismogram_tauzi[so + ir]  =  tauz[o + (i+rec_top_right_var)*Nx + j];
      seismogram_tauxzi[so + ir]  =  tauxz[o + (i+rec_top_right_var)*Nx + j];
  }

}

__kernel void Adj_injSrc(
                         __global float *Avx, __global float *Avz,
                         __global float *Ataux, __global float *Atauz, __global float *Atauxz,
                         __global float *res_vx, __global float * res_vz,
                         __global float *res_taux, __global float *res_tauz, __global float *res_tauxz,
                         int dxr, int ts
                         )


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;
  int s = get_global_id(2);
  int o = s * Nx * Nz;
  int ofs = ts * Ns * Nr + s * Nr;

if(i%dxr==0 && j == rec_top_right_const){
    int ir =  i/dxr;
    Avx[o + (i+rec_top_right_var)*Nx + j] += res_vx[ofs + ir];
    Avz[o + (i+rec_top_right_var)*Nx + j] += res_vz[ofs + ir];
    Ataux[o + (i+rec_top_right_var)*Nx + j] += res_taux[ofs + ir];
    Atauz[o + (i+rec_top_right_var)*Nx + j] += res_tauz[ofs + ir];
    Atauxz[o + (i+rec_top_right_var)*Nx + j] += res_tauxz[ofs + ir];
    
  }

}

__kernel void hessian_seismogram(__global float *vx,__global float *vz,
                     __global float *taux, __global float *tauz, __global float *tauxz,
                     __global float *seismogram_vxi,__global float *seismogram_vzi,
                     __global float *seismogram_tauxi, __global float *seismogram_tauzi, __global float *seismogram_tauxzi,
                     int dxr, int recDepth, int first_rec)


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;

   if(i%dxr==0 && j == recDepth){
    int ir =  i/dxr;
      seismogram_vxi[ir]  =  vx[(i+first_rec)*Nx + j];
      seismogram_vzi[ir]  =  vx[(i+first_rec)*Nx + j];
      seismogram_tauxi[ir]  =  taux[(i+first_rec)*Nx + j];
      seismogram_tauzi[ir]  =  tauz[(i+first_rec)*Nx + j];
      seismogram_tauxzi[ir]  =  tauxz[(i+first_rec)*Nx + j];
  }

}