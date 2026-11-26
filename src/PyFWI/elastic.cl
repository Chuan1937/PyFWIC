
      ////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////
      ////      Make the matrices zero for each source                    ////
      ////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////

__kernel void MakeAllZero(__global float *vx, __global float *vz,
                          __global float *taux,__global float *tauz, __global float *tauxz
                          )
{

  int i = get_global_id(0);
  int j = get_global_id(1);

  int s = get_global_id(2);
  int o = s * Nx * Nz;
    vx[o + center]=0;
    vz[o + center]=0;
    taux[o + center]=0;
    tauz[o + center]=0;
    tauxz[o + center]=0;
}

__kernel void MakeGradZero(__global float *Gmu,__global float *Glam, __global float *Grho,
                          __global float *Gmu_precond, __global float *Glam_precond, __global float *Grho_precond)
{

  int i = get_global_id(0);
  int j = get_global_id(1);

  Gmu[center]=0;
  Glam[center]=0;
  Grho[center]=0;
#if ENERGY_BALANCING
  Gmu_precond[center]=0;
  Glam_precond[center]=0;
  Grho_precond[center]=0;
#endif
}

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////                        source injection                        ////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
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
  
  if(i%dxr==0 && j == rec_top_left_const && i/dxr<n_extera_rec){
    int ir = i/dxr;    

      seismogram_vxi[so + ir]  =  vx[o + (i+ rec_top_left_var)*Nx + j];
      seismogram_vzi[so + ir]  =  vz[o + (i+ rec_top_left_var)*Nx + j];
      seismogram_tauxi[so + ir]  =  taux[o + (i+ rec_top_left_var)*Nx + j];
      seismogram_tauzi[so + ir]  =  tauz[o + (i+ rec_top_left_var)*Nx + j];
      seismogram_tauxzi[so + ir]  =  tauxz[o + (i+ rec_top_left_var)*Nx + j];
      
  }

  if(j%dxr==0 && i == rec_surface_const){
    int ir =  j/dxr;
    if (ir < n_main_rec){
        seismogram_vxi[so + n_extera_rec + ir]  =  vx[o + i*Nx + (j + rec_surface_var)];
        seismogram_vzi[so + n_extera_rec + ir]  =  vz[o + i*Nx + (j + rec_surface_var)];
        seismogram_tauxi[so + n_extera_rec + ir]  =  taux[o + i*Nx + (j + rec_surface_var)];
        seismogram_tauzi[so + n_extera_rec + ir]  =  tauz[o + i*Nx + (j + rec_surface_var)];
        seismogram_tauxzi[so + n_extera_rec + ir]  =  tauxz[o + i*Nx + (j + rec_surface_var)];
    }
  }

  int startOfThirdRec = n_extera_rec + n_main_rec;

  if(i%dxr==0 && j == rec_top_right_const && i/dxr<n_extera_rec){
    int ir = i/dxr;
      seismogram_vxi[so + startOfThirdRec + ir]  =  vx[o + (i+ rec_top_right_var)*Nx + j];
      seismogram_vzi[so + startOfThirdRec + ir]  =  vz[o + (i+ rec_top_right_var)*Nx + j];
      seismogram_tauxi[so + startOfThirdRec + ir]  =  taux[o + (i+ rec_top_right_var)*Nx + j];
      seismogram_tauzi[so + startOfThirdRec + ir]  =  tauz[o + (i+ rec_top_right_var)*Nx + j];
      seismogram_tauxzi[so + startOfThirdRec + ir]  =  tauxz[o + (i+ rec_top_right_var)*Nx + j];
  }

}


__kernel void forward_hessian_src_preparing(__global float *w_vx,__global float *w_vz,
                     __global float *w_taux, __global float *w_tauz, __global float *w_tauxz,
                     __global float *vx,__global float *vz,
                     __global float *taux, __global float *tauz, __global float *tauxz,
                     __global float *glam, __global float *gmu, __global float *grho,
                     __global float *rho
                     )


{
  int i = get_global_id(0) ;
  int j = get_global_id(1) ;

  if(i<sdo || j<sdo || i>=Nz-sdo || j>=Nx-sdo){return;}

  float DxmTx = (c1*(w_taux[center]-w_taux[left]) + c2*(w_taux[right]-w_taux[left2])+
               c3*(w_taux[right2]-w_taux[left3]) + c4*(w_taux[right3]-w_taux[left4]))/dx ;

  float DzmTxz = (c1*(w_tauxz[center]-w_tauxz[above]) + c2*(w_tauxz[below]-w_tauxz[above2])+
               c3*(w_tauxz[below2]-w_tauxz[above3]) + c4*(w_tauxz[below3]-w_tauxz[above4]))/dz;

  float DzpTz = (c1*(w_tauz[below]-w_tauz[center]) + c2*(w_tauz[below2]-w_tauz[above])+
               c3*(w_tauz[below3]-w_tauz[above2]) + c4*(w_tauz[below4]-w_tauz[above3]))/dz;

  float DxpTxz = (c1*(w_tauxz[right]-w_tauxz[center]) + c2*(w_tauxz[right2]-w_tauxz[left])+
               c3*(w_tauxz[right3]-w_tauxz[left2]) + c4*(w_tauxz[right4]-w_tauxz[left3]))/dx;

  float DzmVz= (c1*(w_vz[center]-w_vz[above]) + c2*(w_vz[below]-w_vz[above2])+
               c3*(w_vz[below2]-w_vz[above3]) + c4*(w_vz[below3]-w_vz[above4]))/dz; 

  float DxpVx= (c1*(w_vx[right]-w_vx[center]) + c2*(w_vx[right2]-w_vx[left])+
               c3*(w_vx[right3]-w_vx[left2]) + c4*(w_vx[right4]-w_vx[left3]))/dx;

  float DzpVx= (c1*(w_vz[below]-w_vz[center]) + c2*(w_vz[below2]-w_vz[above])+
               c3*(w_vz[below3]-w_vz[above2]) + c4*(w_vz[below4]-w_vz[above3]))/dz;

  float DxmVz= (c1*(w_vz[center]-w_vz[left]) + c2*(w_vz[right]-w_vz[left2])+
               c3*(w_vz[right2]-w_vz[left3]) + c4*(w_vz[right3]-w_vz[left4]))/dx;


  vx[center] += - dt * (grho[center] * (rho[center] * rho[center] * rho[center])) * (DxmTx + DzmTxz);
  vz[center] += - dt * (grho[center] * (rho[center] * rho[center] * rho[center])) * (DzpTz + DxpTxz);
  taux[center] += dt * glam[center] * (DxpVx + DzmVz) + 2 * dt * gmu[center] * DxpVx;
  tauz[center] += dt * glam[center] * (DxpVx + DzmVz) + 2 * dt * gmu[center] * DzmVz;
  tauxz[center] += dt * gmu[center] * (DzpVx + DxmVz);
  
}

/////////////////////////////// Updating for forward modelling ///////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////////                 Update velocities  in x-direction      ////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

__kernel void update_velx(int coeff,
             __global float *vx,
            __global float *taux,__global float *tauxz,
            __global float *rho_b,
            __global float *dxpml, __global float *dzpml
            )

{
  int i = get_global_id(1)+sdo ;
  int j = get_global_id(0)+sdo ;

  if(i>=Nz-sdo || j>=Nx-sdo){return;}

  float DxmTx = (c1*(taux[center]-taux[left]) + c2*(taux[right]-taux[left2])+
               c3*(taux[right2]-taux[left3]) + c4*(taux[right3]-taux[left4]))/dx ;

  float DzmTxz = (c1*(tauxz[center]-tauxz[above]) + c2*(tauxz[below]-tauxz[above2])+
               c3*(tauxz[below2]-tauxz[above3]) + c4*(tauxz[below3]-tauxz[above4]))/dz;


    if (coeff==1){
      vx[center] += coeff * dt * (rho_b[center] * (DxmTx + DzmTxz));
      vx[center] += coeff * dt * (- (dxpml[center] + dzpml[center])*vx[center]);

    }else if(coeff==-1){
      vx[center] += coeff * dt * (+ (dxpml[center] + dzpml[center])*vx[center]);
      vx[center] += coeff * dt * (rho_b[center] * (DxmTx + DzmTxz));
    
    } 

}

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////////                 Update velocities  in z-direction      ////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

__kernel void update_velz(int coeff,
            __global float *vz,
            __global float *tauz,__global float *tauxz,
            __global float *rho_b,
            __global float *dxpml, __global float *dzpml
            )

{
  int i = get_global_id(1)+sdo ;
  int j = get_global_id(0)+sdo ;

  if(i>=Nz-sdo || j>=Nx-sdo){return;}



  float DxpTxz= (c1*(tauxz[right]-tauxz[center]) + c2*(tauxz[right2]-tauxz[left])+
               c3*(tauxz[right3]-tauxz[left2]) + c4*(tauxz[right4]-tauxz[left3]))/dx;


  float DzpTz= (c1*(tauz[below]-tauz[center]) + c2*(tauz[below2]-tauz[above])+
               c3*(tauz[below3]-tauz[above2]) + c4*(tauz[below4]-tauz[above3]))/dz;


if (coeff==1){
      vz[center] += coeff * dt * (rho_b[center] * (DzpTz + DxpTxz));
      vz[center] += coeff * dt * (-(dzpml[center] + dxpml[center])*vz[center]);

    }else if(coeff==-1){
      vz[center] += coeff * dt * (+(dzpml[center] + dxpml[center])*vz[center]);
      vz[center] += coeff * dt * (rho_b[center] * (DzpTz + DxpTxz));
    } 


}


        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////////                 Update pressure  in z-direction      ////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

__kernel void update_tauz(int coeff,
            __global float *vx, __global float *vz,
            __global float *taux, __global float *tauz,
            __global float *lambda, __global float *mu,
            __global float *dxpml, __global float *dzpml
            )

{
  int i = get_global_id(1)+sdo ;
  int j = get_global_id(0)+sdo ;

  if(i>=Nz-sdo || j>=Nx-sdo){return;}

  float DxpVx= (c1*(vx[right]-vx[center]) + c2*(vx[right2]-vx[left])+
               c3*(vx[right3]-vx[left2]) + c4*(vx[right4]-vx[left3]))/dx;

  float DzmVz= (c1*(vz[center]-vz[above]) + c2*(vz[below]-vz[above2])+
               c3*(vz[below2]-vz[above3]) + c4*(vz[below3]-vz[above4]))/dz;


  if (coeff==1){
      taux[center]+= coeff* dt * ((lambda[center] + 2 * mu[center]) * DxpVx + lambda[center] * DzmVz);
      tauz[center]+= coeff* dt * ((lambda[center] + 2 * mu[center]) * DzmVz + lambda[center] * DxpVx);

      taux[center] += coeff * dt * (-(dxpml[center]+dzpml[center])*taux[center]);
      tauz[center] += coeff * dt * (-(dxpml[center]+dzpml[center])*tauz[center]);

    }else if(coeff==-1){
      taux[center] += coeff * dt * (+(dxpml[center]+dzpml[center])*taux[center]);
      tauz[center] += coeff * dt * (+(dxpml[center]+dzpml[center])*tauz[center]);

      taux[center]+= coeff* dt * ((lambda[center] + 2 * mu[center]) * DxpVx + lambda[center] * DzmVz);
      tauz[center]+= coeff* dt * ((lambda[center] + 2 * mu[center]) * DzmVz + lambda[center] * DxpVx);
    } 

}
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        ////////                 Update pressure  in xz-direction      ////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

__kernel void update_tauxz(int coeff,
            __global float *vx, __global float *vz,
            __global float *tauxz,
            __global float *mu,
            __global float *dxpml, __global float *dzpml
            )

{
  int i = get_global_id(1)+sdo ;
  int j = get_global_id(0)+sdo ;

  if(i>=Nz-sdo || j>=Nx-sdo){return;}


  float DxmVz= (c1*(vz[center]-vz[left]) + c2*(vz[right]-vz[left2])+
               c3*(vz[right2]-vz[left3]) + c4*(vz[right3]-vz[left4]))/dx;

  float DzpVx= (c1*(vx[below]-vx[center]) + c2*(vx[below2]-vx[above])+
               c3*(vx[below3]-vx[above2]) + c4*(vx[below4]-vx[above3]))/dz;



  if (coeff==1){
      tauxz[center] += coeff * dt * (mu[center] *(DxmVz + DzpVx));
      tauxz[center] += coeff * dt * (-(dxpml[center] + dzpml[center])*tauxz[center]);

    }else if(coeff==-1){
      tauxz[center] += coeff * dt * (+(dxpml[center] + dzpml[center])*tauxz[center]);
      tauxz[center] += coeff * dt * (mu[center] *(DxmVz + DzpVx));
    }
    
}


        ////////////////////////////////////////////////////////////////////////
        // Fused updates: same arithmetic and ordering as update_velx/velz and
        // update_tauz/tauxz, with half the kernel launches per time step.
        ////////////////////////////////////////////////////////////////////////
__kernel void update_vel(int coeff,
            __global float *vx, __global float *vz,
            __global float *taux, __global float *tauz, __global float *tauxz,
            __global float *rho_b,
            __global float *dxpml, __global float *dzpml
            )

{
  int i = get_global_id(1)+sdo ;
  int j = get_global_id(0)+sdo ;

  if(i>=Nz-sdo || j>=Nx-sdo){return;}

  int s = get_global_id(2);
  int o = s * Nx * Nz;

    float DxmTx = (c1*(taux[o+center]-taux[o+left]) + c2*(taux[o+right]-taux[o+left2])+
                 c3*(taux[o+right2]-taux[o+left3]) + c4*(taux[o+right3]-taux[o+left4]))/dx ;

    float DzmTxz = (c1*(tauxz[o+center]-tauxz[o+above]) + c2*(tauxz[o+below]-tauxz[o+above2])+
                 c3*(tauxz[o+below2]-tauxz[o+above3]) + c4*(tauxz[o+below3]-tauxz[o+above4]))/dz;

    float DxpTxz= (c1*(tauxz[o+right]-tauxz[o+center]) + c2*(tauxz[o+right2]-tauxz[o+left])+
                 c3*(tauxz[o+right3]-tauxz[o+left2]) + c4*(tauxz[o+right4]-tauxz[o+left3]))/dx;

    float DzpTz= (c1*(tauz[o+below]-tauz[o+center]) + c2*(tauz[o+below2]-tauz[o+above])+
                 c3*(tauz[o+below3]-tauz[o+above2]) + c4*(tauz[o+below4]-tauz[o+above3]))/dz;

    if (coeff==1){
        vx[o+center] += coeff * dt * (rho_b[center] * (DxmTx + DzmTxz));
        vx[o+center] += coeff * dt * (- (dxpml[center] + dzpml[center])*vx[o+center]);

        vz[o+center] += coeff * dt * (rho_b[center] * (DzpTz + DxpTxz));
        vz[o+center] += coeff * dt * (-(dzpml[center] + dxpml[center])*vz[o+center]);

      }else if(coeff==-1){
        vx[o+center] += coeff * dt * (+ (dxpml[center] + dzpml[center])*vx[o+center]);
        vx[o+center] += coeff * dt * (rho_b[center] * (DxmTx + DzmTxz));

        vz[o+center] += coeff * dt * (+(dzpml[center] + dxpml[center])*vz[o+center]);
        vz[o+center] += coeff * dt * (rho_b[center] * (DzpTz + DxpTxz));
      }

}

__kernel void update_tau(int coeff,
            __global float *vx, __global float *vz,
            __global float *taux, __global float *tauz, __global float *tauxz,
            __global float *lambda, __global float *mu,
            __global float *dxpml, __global float *dzpml
            )

{
  int i = get_global_id(1)+sdo ;
  int j = get_global_id(0)+sdo ;

  if(i>=Nz-sdo || j>=Nx-sdo){return;}

  int s = get_global_id(2);
  int o = s * Nx * Nz;

    float DxpVx= (c1*(vx[o+right]-vx[o+center]) + c2*(vx[o+right2]-vx[o+left])+
                 c3*(vx[o+right3]-vx[o+left2]) + c4*(vx[o+right4]-vx[o+left3]))/dx;

    float DzmVz= (c1*(vz[o+center]-vz[o+above]) + c2*(vz[o+below]-vz[o+above2])+
                 c3*(vz[o+below2]-vz[o+above3]) + c4*(vz[o+below3]-vz[o+above4]))/dz;

    float DxmVz= (c1*(vz[o+center]-vz[o+left]) + c2*(vz[o+right]-vz[o+left2])+
                 c3*(vz[o+right2]-vz[o+left3]) + c4*(vz[o+right3]-vz[o+left4]))/dx;

    float DzpVx= (c1*(vx[o+below]-vx[o+center]) + c2*(vx[o+below2]-vx[o+above])+
                 c3*(vx[o+below3]-vx[o+above2]) + c4*(vx[o+below4]-vx[o+above3]))/dz;

    if (coeff==1){
        taux[o+center]+= coeff* dt * ((lambda[center] + 2 * mu[center]) * DxpVx + lambda[center] * DzmVz);
        tauz[o+center]+= coeff* dt * ((lambda[center] + 2 * mu[center]) * DzmVz + lambda[center] * DxpVx);

        taux[o+center] += coeff * dt * (-(dxpml[center]+dzpml[center])*taux[o+center]);
        tauz[o+center] += coeff * dt * (-(dxpml[center]+dzpml[center])*tauz[o+center]);

        tauxz[o+center] += coeff * dt * (mu[center] *(DxmVz + DzpVx));
        tauxz[o+center] += coeff * dt * (-(dxpml[center] + dzpml[center])*tauxz[o+center]);

      }else if(coeff==-1){
        taux[o+center] += coeff * dt * (+(dxpml[center]+dzpml[center])*taux[o+center]);
        tauz[o+center] += coeff * dt * (+(dxpml[center]+dzpml[center])*tauz[o+center]);

        taux[o+center]+= coeff* dt * ((lambda[center] + 2 * mu[center]) * DxpVx + lambda[center] * DzmVz);
        tauz[o+center]+= coeff* dt * ((lambda[center] + 2 * mu[center]) * DzmVz + lambda[center] * DxpVx);

        tauxz[o+center] += coeff * dt * (+(dxpml[center] + dzpml[center])*tauxz[o+center]);
        tauxz[o+center] += coeff * dt * (mu[center] *(DxmVz + DzpVx));
      }

}

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

  int i = get_global_id(0) ;
  int j = get_global_id(1);
  int startOfThirdRec = n_extera_rec + n_main_rec;

  int s = get_global_id(2);
  int o = s * Nx * Nz;
    int ofs = ts * Ns * Nr + s * Nr;

    if(i%dxr==0 && j == rec_top_left_const && i/dxr<n_extera_rec){
      int ir = i/dxr;
        Avx[o + (i+ rec_top_left_var)*Nx + j] += res_vx[ofs + ir];
        Avz[o + (i+ rec_top_left_var)*Nx + j] += res_vz[ofs + ir];
        Ataux[o + (i+ rec_top_left_var)*Nx + j] += res_taux[ofs + ir];
        Atauz[o + (i+ rec_top_left_var)*Nx + j] += res_tauz[ofs + ir];
        Atauxz[o + (i+ rec_top_left_var)*Nx + j] += res_tauxz[ofs + ir];
    }
    if(j%dxr==0 && i == rec_surface_const){
      int ir =  j/dxr;
      if (ir < n_main_rec){

          Avx[o + i*Nx + (j+ rec_surface_var)] += res_vx[ofs + n_extera_rec + ir];
          Avz[o + i*Nx + (j+ rec_surface_var)] += res_vz[ofs + n_extera_rec + ir];
          Ataux[o + i*Nx + (j+ rec_surface_var)] += res_taux[ofs + n_extera_rec + ir];
          Atauz[o + i*Nx + (j+ rec_surface_var)] += res_tauz[ofs + n_extera_rec + ir];
          Atauxz[o + i*Nx + (j+ rec_surface_var)] += res_tauxz[ofs + n_extera_rec + ir];
          }
    }

    if(i%dxr==0 && j == rec_top_right_const && i/dxr<n_extera_rec){
      int ir = i/dxr;

        Avx[o + (i+ rec_top_right_var)*Nx + j] += res_vx[ofs + startOfThirdRec + ir];
        Avz[o + (i+ rec_top_right_var)*Nx + j] += res_vz[ofs + startOfThirdRec + ir];
        Ataux[o + (i+ rec_top_right_var)*Nx + j] += res_taux[ofs + startOfThirdRec + ir];
        Atauz[o + (i+ rec_top_right_var)*Nx + j] += res_tauz[ofs + startOfThirdRec + ir];
        Atauxz[o + (i+ rec_top_right_var)*Nx + j] += res_tauxz[ofs + startOfThirdRec + ir];
    }

}


        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        //////////         Update Ajoint pressure  in x-direction      /////////
        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

__kernel void Adj_update_tau(__global float *Avx, __global float *Avz,
            __global float *Ataux, __global float *Atauz,__global float *Atauxz,
            __global float *rho_b,
            __global float *dxpml, __global float *dzpml
            )

{
  int i = get_global_id(1)+sdo ;
  int j = get_global_id(0)+sdo ;

  if(i>=Nz-sdo || j>=Nx-sdo){return;}

  int s = get_global_id(2);
  int o = s * Nx * Nz;

    float DxpAVx= (c1*(Avx[o+right]*rho_b[right] -Avx[o+center]*rho_b[center]) +
                   c2*(Avx[o+right2]*rho_b[right2] -Avx[o+left]*rho_b[left])+
                   c3*(Avx[o+right3]*rho_b[right3] -Avx[o+left2]*rho_b[left2]) +
                   c4*(Avx[o+right4]*rho_b[right4] -Avx[o+left3]*rho_b[left3]))/dx;

    float DzmAVz= (c1*(Avz[o+center]*rho_b[center] -Avz[o+above]*rho_b[above]) +
                   c2*(Avz[o+below]*rho_b[below] -Avz[o+above2]*rho_b[above2])+
                 c3*(Avz[o+below2]*rho_b[below2] -Avz[o+above3]*rho_b[above3]) +
                 c4*(Avz[o+below3]*rho_b[below3] -Avz[o+above4]*rho_b[above4]))/dz;

    Ataux[o+center]+= - dt * ((dxpml[center])*Ataux[o+center] + DxpAVx);
    Atauz[o+center]+= - dt * ((dzpml[center])*Atauz[o+center] + DzmAVz);

    float DzpAvx= (c1*(Avx[o+below]*rho_b[below] - Avx[o+center]*rho_b[center]) +
                   c2*(Avx[o+below2]*rho_b[below2] -Avx[o+above]*rho_b[above])+
                   c3*(Avx[o+below3]*rho_b[below3] -Avx[o+above2]*rho_b[above2]) +
                   c4*(Avx[o+below4]*rho_b[below4] -Avx[o+above3]*rho_b[above3]))/dz;

    float DxmAvz= (c1*(Avz[o+center]*rho_b[center] - Avz[o+left]*rho_b[left]) +
                   c2*(Avz[o+right]*rho_b[right] - Avz[o+left2]*rho_b[left2])+
                   c3*(Avz[o+right2]*rho_b[right2] - Avz[o+left3]*rho_b[left3]) +
                   c4*(Avz[o+right3]*rho_b[right3] - Avz[o+left4]*rho_b[left4]))/dx;

    Atauxz[o+center]+= - dt*(((dxpml[center] + dzpml[center]) *Atauxz[o+center]) + (DzpAvx +  DxmAvz));

}

__kernel void Adj_update_v(__global float *Avx, __global float *Avz,
            __global float *Ataux, __global float *Atauz, __global float *Atauxz,
            __global float *lam, __global float *mu,
            __global float *dxpml,__global float *dzpml
            )

{
  int i = get_global_id(1)+sdo ;
  int j = get_global_id(0)+sdo ;

  if(i>=Nz-sdo || j>=Nx-sdo){return;}

  int s = get_global_id(2);
  int o = s * Nx * Nz;

    float DxmAtaux =  (c1*((lam[center]+2*mu[center])*Ataux[o+center] - (lam[left]+2*mu[left])*Ataux[o+left]) +
                         c2*((lam[right]+2*mu[right])*Ataux[o+right] - (lam[left2]+2*mu[left2])*Ataux[o+left2])+
                        c3*((lam[right2]+2*mu[right2])*Ataux[o+right2] - (lam[left3]+2*mu[left3])*Ataux[o+left3]) +
                         c4*((lam[right3]+2*mu[right3])*Ataux[o+right3] -(lam[left4]+2*mu[left4])*Ataux[o+left4]))/dx;

    float DxmAtauz = (c1*(lam[center]*Atauz[o+center] - lam[left]*Atauz[o+left]) +
                      c2*(lam[right]*Atauz[o+right] - lam[left2]*Atauz[o+left2]) +
                      c3*(lam[right2]*Atauz[o+right2] - lam[left3]*Atauz[o+left3]) +
                      c4*(lam[right3]*Atauz[o+right3] - lam[left4]*Atauz[o+left4]))/dx;

    float DzmAtauxz=  (c1*(mu[center]*Atauxz[o+center] -mu[above]*Atauxz[o+above]) +
                      c2*(mu[below]*Atauxz[o+below] - mu[above2]*Atauxz[o+above2])+
                      c3*(mu[below2]*Atauxz[o+below2] -mu[above3]*Atauxz[o+above3]) +
                      c4*(mu[below3]*Atauxz[o+below3] - mu[above4]*Atauxz[o+above4]))/dz;

    float DzpAtauz =(c1*((lam[below]+2*mu[below])*Atauz[o+below] -(lam[center]+2*mu[center])*Atauz[o+center]) +
                     c2*((lam[below2]+2*mu[below2])*Atauz[o+below2] -(lam[above]+2*mu[above])*Atauz[o+above])+
                     c3*((lam[below3]+2*mu[below3])*Atauz[o+below3] -(lam[above2]+2*mu[above2])*Atauz[o+above2]) +
                     c4*((lam[below4]+2*mu[below4])*Atauz[o+below4] -(lam[above3]+2*mu[above3])*Atauz[o+above3]))/dz;

    float DzpAtaux = (c1*(lam[below]*Ataux[o+below] -lam[center]*Ataux[o+center]) +
                      c2*(lam[below2]*Ataux[o+below2] -lam[above]*Ataux[o+above])+
                      c3*(lam[below3]*Ataux[o+below3]-lam[above2]*Ataux[o+above2]) +
                      c4*(lam[below4]*Ataux[o+below4]-lam[above3]*Ataux[o+above3]))/dz;

    float DxpAtauxz= (c1*(mu[right] *Atauxz[o+right] - mu[center]*Atauxz[o+center]) +
                      c2*(mu[right2]*Atauxz[o+right2] - mu[left]*Atauxz[o+left])+
                      c3*(mu[right3]*Atauxz[o+right3] - mu[left2]*Atauxz[o+left2]) +
                      c4*(mu[right4]*Atauxz[o+right4] - mu[left3]*Atauxz[o+left3]))/dx;

    Avx[o+center]+= -dt* ((2*dxpml[center] + dzpml[center]) * Avx[o+center] + (DxmAtaux + DxmAtauz + DzmAtauxz));
    Avz[o+center]+= -dt* ((dxpml[center] + 2*dzpml[center]) * Avz[o+center] + (DzpAtaux + DzpAtauz + DxpAtauxz));

}

__kernel void Grad(__global float *vx, __global float *vz,
            __global float *taux, __global float *tauz, __global float *tauxz,
            __global float *Avx, __global float *Avz,
            __global float *Ataux, __global float *Atauz,  __global float *Atauxz,
            __global float *lambda, __global float *mu, __global float *rho_b,
            __global float *Gmu, __global float *Gmu_precond,
            __global float *Glam, __global float *Glam_precond,
            __global float *Grho, __global float *Grho_precond
            )
{
  int i = get_global_id(1)+ npml + sdo ;
  int j = get_global_id(0)+ npml + sdo ;

  if(i>=Nz-sdo - npml || j>=Nx-sdo - npml){return;}

  for (int s = 0; s < Ns; ++s) {
    int o = s * Nx * Nz;

    float DzpVx= (c1*(vz[o+below]-vz[o+center]) + c2*(vz[o+below2]-vz[o+above])+
                 c3*(vz[o+below3]-vz[o+above2]) + c4*(vz[o+below4]-vz[o+above3]))/dz;

    float DxmVz= (c1*(vz[o+center]-vz[o+left]) + c2*(vz[o+right]-vz[o+left2])+
                 c3*(vz[o+right2]-vz[o+left3]) + c4*(vz[o+right3]-vz[o+left4]))/dx;

    float DxpVx= (c1*(vx[o+right]-vx[o+center]) + c2*(vx[o+right2]-vx[o+left])+
                 c3*(vx[o+right3]-vx[o+left2]) + c4*(vx[o+right4]-vx[o+left3]))/dx;

    float DzmVz= (c1*(vz[o+center]-vz[o+above]) + c2*(vz[o+below]-vz[o+above2])+
                 c3*(vz[o+below2]-vz[o+above3]) + c4*(vz[o+below3]-vz[o+above4]))/dz;

    float DzpPz= (c1*(tauz[o+below]-tauz[o+center]) + c2*(tauz[o+below2]-tauz[o+above])+
                 c3*(tauz[o+below3]-tauz[o+above2]) + c4*(tauz[o+below4]-tauz[o+above3]))/dz;

    float DxmPx= (c1*(taux[o+center]-taux[o+left]) + c2*(taux[o+right]-taux[o+left2])+
                 c3*(taux[o+right2]-taux[o+left3]) + c4*(taux[o+right3]-taux[o+left4]))/dx;

    float DxpPxz= (c1*(tauxz[o+right]-tauxz[o+center]) + c2*(tauxz[o+right2]-tauxz[o+left])+
                  c3*(tauxz[o+right3]-tauxz[o+left2]) + c4*(tauxz[o+right4]-tauxz[o+left3]))/dx;

    float DzmPxz= (c1*(tauxz[o+center]-tauxz[o+above]) + c2*(tauxz[o+below]-tauxz[o+above2])+
                  c3*(tauxz[o+below2]-tauxz[o+above3]) + c4*(tauxz[o+below3]-tauxz[o+above4]))/dz;

    Gmu[center]+= 2*dt*(Ataux[o+center]*DxpVx + Atauz[o+center]*DzmVz) + dt*Atauxz[o+center]*(DzpVx+DxmVz);

#if ENERGY_BALANCING
    Gmu_precond[center]+= 4*dt*dt*(taux[o+center]*taux[o+center]*DxpVx*DxpVx + tauz[o+center]*tauz[o+center]*DzmVz*DzmVz)
                          +dt*dt*tauxz[o+center]*tauxz[o+center]*(DzpVx*DzpVx+DxmVz*DxmVz);
#endif

    Glam[center]+= dt*(Ataux[o+center] + Atauz[o+center])*(DxpVx+DzmVz);

#if ENERGY_BALANCING
    Glam_precond[center] += dt*dt*(taux[o+center]*taux[o+center] + tauz[o+center]*tauz[o+center])*(DxpVx*DxpVx+DzmVz*DzmVz);
#endif

    Grho[center]+= -(dt*(rho_b[center]*rho_b[center]))*(Avx[o+center]*(DxmPx + DzmPxz)+
                                                    Avz[o+center]*(DzpPz+DxpPxz));

#if ENERGY_BALANCING
    Grho_precond[center] += dt*dt*(vx[o+center]*vx[o+center]*(DxmPx*DxmPx + DzmPxz*DzmPxz)+
                                              vz[o+center]*vz[o+center]*(DzpPz*DzpPz + DxpPxz*DxpPxz));
#endif
  }

}
