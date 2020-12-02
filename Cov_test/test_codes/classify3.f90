      subroutine classify3(natom,Rc,nk_8,num_neigh, &
        list_neigh,dR_neigh,iat_neigh,ntype,m_neigh,iatom,iat_type, &
        iat_class,nbond,nangle, &
        value1_ave,value11_ave,num1_ave, &
        prob,prob_dist,iflag_run, &
        num_cluster,ind_cluster,iatom_cluster,featk,feat_cluster, &
        bond_dist,theta_dist)
        

!ccccc This code will not use smooth feature, instead will assign each
!case to one grid point
      use mod_bond_angle
      implicit none
      integer natom,ntype
      integer nfeat0,m_neigh
      real*8 Rc,Rc2
      real*8 dR_neigh(3,m_neigh,ntype,natom)
      real*8 dR_neigh_alltype(3,m_neigh,natom)
      integer iat_neigh(m_neigh,ntype,natom),list_neigh(m_neigh,ntype,natom)
      integer num_neigh(ntype,natom)
      integer num_neigh_alltype(natom)
      integer nperiod(3)
      integer iflag,i,j,num,iat,itype
      integer i1,i2,i3,itype1,itype2,j1,j2,iat1,iat2
      real*8 d,dx1,dx2,dx3,dx,dy,dz,dd
      real*8 pi,pi2,x,f1
      integer nk1,nk2
      integer iatom(natom),iat_type(ntype)
      real*8 bond_dist(1000,ntype),theta_dist(1000,ntype,ntype)

      integer inf_f32(2),k,k1,k2,k12,j12,ii_f,jj,jj1,jj2,nneigh
      real*8 y,y2
      integer itype12,ind_f32(2)
      integer id,itheta

      integer nk_8(8)
      real*8 prob_dist(100,8)
      real*8 prob(nk_8(1),nk_8(2),nk_8(3),nk_8(4),nk_8(5),nk_8(6),nk_8(7),nk_8(8))
      integer, allocatable, dimension (:) :: ik1,ik2
      real*8 value1_ave(10),value11_ave(10)
      integer num1_ave(10)
      integer iflag_run

!      integer itype_bond(10),nxp_bond(10),nk_bond(10)
!      integer nxp_angle(10),nk_angle(10)
!      real*8 scale_bond(10),scale_angle(10)
!      real*8 theta_angle(10),dtheta_angle(10)


      integer iat_class
      real*8 sum_bond(10),sum_angle(10)
      integer ik(10)
      real*8  featk(10,natom)
      real*8  feat_cluster(10,200)
      integer nbond,nangle,ii
      real*8 dd1,d1,dd2,d2,x1,x2,d12,theta,fact1,fact2
      real*8 theta1,sum
      integer ind_cluster(10,200),num_cluster
      integer iatom_cluster(natom)
      real*8 amin
      integer iimin


      pi=4*datan(1.d0)
      pi2=2*pi

      featk=0.d0

      do 1000 iat=1,natom

      if(iatom(iat).ne.iat_class) goto 1000

      ik(:)=1

      sum_bond=0.d0
      do  itype=1,ntype
      do   j=1,num_neigh(itype,iat)
      dd=dR_neigh(1,j,itype,iat)**2+dR_neigh(2,j,itype,iat)**2+dR_neigh(3,j,itype,iat)**2
      d=dsqrt(dd)
       id=int(d/Rc*(1000.d0-2))+1
       bond_dist(id,itype)=bond_dist(id,itype)+1
       
       do ii=1,nbond
       if(itype.eq.itype_bond(ii)) then
       if(d.lt.Rc_bond(ii)) then
       x=d/Rc_bond(ii)*pi/2
       sum_bond(ii)=sum_bond(ii)+(1.d0-sin(x)**nxp_bond(ii))*(XX_bond(ii)+ &
           (1-XX_bond(ii))*exp(-(d-Rg_bond(ii))**2/dRg_bond(ii)**2))
       endif
       endif
       enddo
      enddo
      enddo

!ccccccccccccccccccccccccccccccccccccccc
      do ii=1,nbond

      if(iflag_run.eq.1) then
      value1_ave(ii)=value1_ave(ii)+sum_bond(ii)
      value11_ave(ii)=value11_ave(ii)+sum_bond(ii)**2
      num1_ave(ii)=num1_ave(ii)+1
      else
      sum_bond(ii)=(sum_bond(ii)-value1_ave(ii))*value11_ave(ii)
      sum_bond(ii)=sum_bond(ii)*scale_bond(ii)+nk_bond(ii)/2.d0
      endif

      ik(ii)=int(sum_bond(ii))+1
      featk(ii,iat)=sum_bond(ii)
      if(ik(ii).gt.nk_bond(ii)) ik(ii)=nk_bond(ii)
      if(ik(ii).lt.1) ik(ii)=1
      prob_dist(ik(ii),ii)=prob_dist(ik(ii),ii)+1
      enddo
!cccccccccccccccccccccccccccccccccccccccccccccccc

      sum_angle=0.d0
      do itype1=1,ntype
      do itype2=1,itype1

      do j1=1,num_neigh(itype1,iat)
      do j2=1,num_neigh(itype2,iat)

      if(itype1.eq.itype2.and.j2.ge.j1) goto 1001


      dd1=dR_neigh(1,j1,itype1,iat)**2+dR_neigh(2,j1,itype1,iat)**2+dR_neigh(3,j1,itype1,iat)**2
      d1=dsqrt(dd1)
      dd2=dR_neigh(1,j2,itype2,iat)**2+dR_neigh(2,j2,itype2,iat)**2+dR_neigh(3,j2,itype2,iat)**2
      d2=dsqrt(dd2)

      d12=dR_neigh(1,j1,itype1,iat)*dR_neigh(1,j2,itype2,iat)+ &
          dR_neigh(2,j1,itype1,iat)*dR_neigh(2,j2,itype2,iat)+ &
          dR_neigh(3,j1,itype1,iat)*dR_neigh(3,j2,itype2,iat)


      theta=d12/(d1*d2)
      if(abs(theta).gt.1.d0-1.E-5) theta=theta/abs(theta)*(1.d0-1.E-5)
      theta=dacos(theta)
      theta=theta*180.d0/pi

      itheta=theta/180.d0*(1000-2)+1
      theta_dist(itheta,itype1,itype2)=theta_dist(itheta,itype1,itype2)+1

      do ii=1,nangle

      iflag=0
      if(itype1.eq.itype1_angle(ii).and.itype2.eq.itype2_angle(ii)) iflag=1
      if(itype2.eq.itype1_angle(ii).and.itype1.eq.itype2_angle(ii)) iflag=1
      if(itype1.eq.itype1_angle(ii).and.itype2_angle(ii).eq.0) iflag=1
      if(itype2.eq.itype1_angle(ii).and.itype2_angle(ii).eq.0) iflag=1
      if(itype1.eq.itype2_angle(ii).and.itype1_angle(ii).eq.0) iflag=1
      if(itype2.eq.itype2_angle(ii).and.itype1_angle(ii).eq.0) iflag=1
      if(itype1_angle(ii).eq.0.and.itype2_angle(ii).eq.0) iflag=1
      if(d1.gt.Rc_angle(ii).or.d2.gt.Rc_angle(ii)) iflag=0

      if(iflag.eq.1) then
      x1=d1/Rc_angle(ii)*pi/2
      x2=d2/Rc_angle(ii)*pi/2

      fact1=(1.d0-sin(x1)**nxp_angle(ii))
      fact2=(1.d0-sin(x2)**nxp_angle(ii))
      sum_angle(ii)=sum_angle(ii)+exp(-(theta-theta_angle(ii))**2/dtheta_angle(ii)**2)*fact1*fact2
      endif
      enddo


1001  continue

      enddo
      enddo
      enddo
      enddo
!ccccccccccccccccccccccccccccccccccc
      do ii=1,nangle

      if(iflag_run.eq.1) then
      value1_ave(ii+nbond)=value1_ave(ii+nbond)+sum_angle(ii)
      value11_ave(ii+nbond)=value11_ave(ii+nbond)+sum_angle(ii)**2
      num1_ave(ii+nbond)=num1_ave(ii+nbond)+1
      else
      sum_angle(ii)=(sum_angle(ii)-value1_ave(ii+nbond))*value11_ave(ii+nbond)
      sum_angle(ii)=sum_angle(ii)*scale_angle(ii)+nk_angle(ii)/2.d0
      endif

      ik(ii+nbond)=int(sum_angle(ii))
      featk(ii+nbond,iat)=sum_angle(ii)
      if(ik(ii+nbond).gt.nk_8(ii+nbond)) ik(ii+nbond)=nk_8(ii+nbond)
      if(ik(ii+nbond).lt.1) ik(ii+nbond)=1
      prob_dist(ik(ii+nbond),ii+nbond)=prob_dist(ik(ii+nbond),ii+nbond)+1
      enddo
!cccccccccccccccccccccccccccccccccccccccccccccccc

      if(iflag_run.eq.2) then
      prob(ik(1),ik(2),ik(3),ik(4),ik(5),ik(6),ik(7),ik(8))= &
          prob(ik(1),ik(2),ik(3),ik(4),ik(5),ik(6),ik(7),ik(8))+1
      endif

      if(iflag_run.eq.3) then
        amin=1.E+20
       do ii=1,num_cluster
         sum=0
         do i=1,8
!         sum=sum+abs(ik(i)-ind_cluster(i,ii))**2
         sum=sum+(featk(i,iat)-feat_cluster(i,ii))**2
         enddo

         if(sum.lt.amin) then
         amin=sum
         iimin=ii
         endif
!         if(sum.lt.0.001) then
!         iatom_cluster(iat)=ii
!         endif
       enddo
       iatom_cluster(iat)=iimin
      endif


1000  continue


      return
      end subroutine classify3
     
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

