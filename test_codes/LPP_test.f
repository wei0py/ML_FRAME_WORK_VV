       program LPP_test
       implicit double precision (a-h,o-z)
       integer(4) :: i
       real*8,allocatable,dimension(:,:) :: feat_case
       real*8,allocatable,dimension(:,:) :: feat2_case
       real*8,allocatable,dimension(:,:) :: feat2_case_tmp
       real*8,allocatable,dimension(:) :: Ei_case
       real*8,allocatable,dimension(:) :: weight_feat
       real*8,allocatable,dimension(:,:) :: W_LPP,D_LPP,L_LPP,left1,
     &   left,right1,right
       real*8,allocatable,dimension(:,:) :: W_LPP2,D_LPP2,L_LPP2
       real*8,allocatable,dimension(:) :: alphaR,alphaI,beta
       real*8,allocatable,dimension(:) :: W_1D
       integer,allocatable,dimension(:) :: imin
       integer :: numVR
       real*8,allocatable,dimension(:,:) :: S
       real*8,allocatable,dimension(:,:) :: PV,PV0
       real*8,allocatable,dimension(:) :: EW,work,BB
       real*8,allocatable,dimension(:) :: BB2_ave
       real*8,allocatable,dimension(:,:) :: BB_store
       real*8,allocatable,dimension(:) :: weight_case
       real*8,allocatable,dimension(:) :: E_fit
       real*8,allocatable,dimension(:) :: feat2_shift,feat2_scale
       real*8,allocatable,dimension(:,:) :: feat_tmp
       real*8,allocatable,dimension(:) :: weight_tmp
       integer,allocatable,dimension(:) :: index


       integer,allocatable,dimension(:) :: ipiv
       integer iatom_type(10)
       real*8 Ei_tmp,sum,epsilon,t_lpp
       integer itmp,lwork
       character(len=80),allocatable,dimension (:) :: trainSetFileDir
       character(len=80) trainSetDir,BadImageDir
       character(len=90) MOVEMENTDir,dfeatDir,infoDir,trainDataDir
       integer sys_num,sys
       
       open(10,file="LPP.input") 
       rewind(10)
       read(10,*) ntype,numVR
       do i=1,ntype
          read(10,*)iatom_type(i)
       end do
       close(10)


         write(6,*) "ntype=", ntype, " input itype"
         read(5,*) itype
         write(6,*) "input imth(1:old;2:new)"
         read(5,*) imth



         open(10,file="feat_new_stored."//char(itype+48),
     &    form="unformatted")
        rewind(10)
        read(10) ncase_tmp,nfeat
         write(6,*) "ncase=", ncase_tmp, " input n_interval"
         read(5,*) ninterv
         

         ncase=ncase_tmp/ninterv
 
       nfeat=nfeat-1   ! last feature is not used
       allocate(feat_case(nfeat,ncase))
       allocate(Ei_case(ncase))
       allocate(weight_feat(nfeat))
       
          icase=0
        do ii=1,ncase
          icase=icase+1
          read(10) iit,Ei_case(icase),feat_case(:,icase)
          do jj=1,ninterv-1
          read(10) 
          enddo
        enddo

        close(10)
       close(10)

         write(6,*)  
     & "input: fact(>1);  ikern(1:exp;2:Rcut;3:neigh); weight power mm"
         write(6,*) 
     & "fact (num neigh,etc),if no weight_feat.type, mm=0"  
     
         read(5,*) fact_lpp,ikern, mm

         if(mm.ne.0) then
         open(10,file="weight_feat."//char(itype+48))
         rewind(10)
         do ii=1,nfeat
         read(10,*) iit,weight_feat(ii)
         enddo
         close(10)

         do jj=1,nfeat
         weight_feat(jj)=dsqrt(abs(weight_feat(jj)))
         enddo

         do ii=1,ncase
         do jj=1,nfeat
         feat_case(jj,ii)=feat_case(jj,ii)*weight_feat(jj)**mm
         enddo
         enddo
         endif

ccccccccccccccccccccccc clean up the feature
         allocate(feat_tmp(nfeat,ncase))
         allocate(weight_tmp(nfeat))
         allocate(index(nfeat))
         jj1=0
         do jj=1,nfeat
         sum=0.d0
         do ii=1,ncase
         sum=sum+abs(feat_case(jj,ii))  ! bad access approach
         enddo
         if(sum.gt.1.D-8) then
         jj1=jj1+1
         index(jj1)=jj
         feat_tmp(jj1,:)=feat_case(jj,:)
         weight_tmp(jj1)=weight_feat(jj)
         endif
         enddo
         write(6,*) "original nfeat=",nfeat
         nfeat0=nfeat
         nfeat=jj1
         write(6,*) "reduced nfeat=",nfeat
         deallocate(feat_case)
         deallocate(weight_feat)
         allocate(feat_case(nfeat,ncase))
         allocate(weight_feat(nfeat))
         do ii=1,ncase
         do jj=1,nfeat
         feat_case(jj,ii)=feat_tmp(jj,ii)
         enddo
         enddo
         do jj=1,nfeat
         weight_feat(jj)=weight_tmp(jj)
         enddo
         deallocate(feat_tmp)
         deallocate(weight_tmp)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         

       allocate(W_LPP(ncase,ncase))
       allocate(D_LPP(ncase,ncase))
       allocate(L_LPP(ncase,ncase))
       allocate(W_1D(ncase))
       allocate(imin(ncase))

       allocate(W_LPP2(ncase,ncase))
       allocate(D_LPP2(ncase,ncase))
       allocate(L_LPP2(ncase,ncase))

       sum=0.d0
       num=0
       do i1=1,ncase/10
       do i2=1,i1 
       do j=1,nfeat
            sum=sum+(feat_case(j,i1)-feat_case(j,i2))**2
       enddo
       num=num+1
       enddo
       enddo

       sum=sum/num    ! this is the average distance

       t_lpp=sum/fact_lpp
       epsilon=sum/fact_lpp

       write(6,*) "epsilon, t_lpp", epsilon, t_lpp

       num_test=0
       do i1=1,ncase
       do i2=1,i1 
       sum=0.d0
       do j=1,nfeat
            sum=sum+(feat_case(j,i1)-feat_case(j,i2))**2
       enddo
         if(sum.lt.epsilon) then
         num_test=num_test+1
          if(i1.ne.i2) num_test=num_test+1
         endif


        if(ikern.eq.1) then
              W_LPP(i1,i2)=exp(-sum/t_lpp)
              W_LPP(i2,i1)=W_LPP(i1,i2)
        elseif(ikern.eq.2) then
            if(sum .lt. epsilon) then
              W_LPP(i1,i2)=1.0
              W_LPP(i2,i1)=1.0
            else
              W_LPP(i1,i2)=0.d0
              W_LPP(i2,i1)=0.d0
            endif
        elseif(ikern.eq.3) then
            W_LPP(i1,i2)=sum
            W_LPP(i2,i1)=sum
        endif

             W_LPP2(i1,i2)=1.0
             W_LPP2(i2,i1)=1.0

       enddo
       enddo

        write(6,*) "average number", num_test*1.d0/ncase

           if(ikern.eq.3) then
          neigh=fact_lpp+1
      do i2=1,ncase
          W_1D(:)=W_LPP(:,i2)
      do ineigh=1,neigh
      amin=1.D+30
      do i1=1,ncase
      if(W_1D(i1).lt.amin) then
      amin=W_1D(i1)
      imin(ineigh)=i1
      endif
      enddo
      W_1D(imin(ineigh))=2.D+30
      enddo
        W_LPP(:,i2)=0.d0
        do ineigh=1,neigh
        W_LPP(imin(ineigh),i2)=1.d0
        enddo
      enddo

      do i1=1,ncase
      do i2=1,i1
      if(W_LPP(i1,i2).gt.0.9.or.W_LPP(i2,i1).gt.0.9) then
      W_LPP(i1,i2)=1.d0
      W_LPP(i2,i1)=1.d0
      endif
      enddo
      enddo
         endif   !ikern=3
      

!         W_LPP2=W_LPP2-W_LPP    ! doesn't has any effect

       D_LPP=0.d0
       D_LPP2=0.d0
       do i1=1,ncase
              sum=0.d0
              sum1=0.d0
              do i=1,ncase
              sum=sum+W_LPP(i,i1)
              sum1=sum1+W_LPP2(i,i1)
              enddo
              D_LPP(i1,i1)=sum
              D_LPP2(i1,i1)=sum1
       enddo


       L_LPP(:,:)=D_LPP(:,:)-W_LPP(:,:)
       L_LPP2(:,:)=D_LPP2(:,:)-W_LPP2(:,:)

       allocate(left1(nfeat,ncase))
       allocate(left(nfeat,nfeat))
       call dgemm('N','N',nfeat,ncase,ncase,1.d0, 
     & feat_case,nfeat,L_LPP,ncase,0.d0,left1,nfeat)

       call dgemm('N','T',nfeat,nfeat,ncase,1.d0,
     &  left1,nfeat,feat_case,nfeat,0.d0,left,nfeat)

       allocate(right1(nfeat,ncase))
       allocate(right(nfeat,nfeat))

       if(imth.eq.1) then
       call dgemm('N','N',nfeat,ncase,ncase,1.d0,
     & feat_case,nfeat,D_LPP,ncase,
     & 0.d0,right1,nfeat)
       elseif(imth.eq.2) then
       call dgemm('N','N',nfeat,ncase,ncase,1.d0,
     & feat_case,nfeat,L_LPP2,ncase,
     & 0.d0,right1,nfeat)
        endif


       call dgemm('N','T',nfeat,nfeat,ncase,1.d0,
     &  right1,nfeat,feat_case,nfeat,0.d0,right,nfeat)

!TODO:
       lwork=10*nfeat
       allocate(work(lwork))
       allocate(EW(nfeat))

!        call sgegv('N','V',nfeat,left,nfeat,right,nfeat,alphaR,alphaI,
!      &  beta,VL,nfeat,VR,nfeat,work,lwork,info)
!        call sggev('N','V',nfeat,left,nfeat,right,nfeat,alphaR,alphaI,
!      &  beta,VL,nfeat,VR,nfeat,work,lwork,info)


         do ii=1,nfeat
      !   right(ii,ii)=right(ii,ii)+1.E-5
         enddo


!       call ssygv(1,'V','U',nfeat,left,nfeat,right,nfeat,
       call dsygv(1,'V','U',nfeat,left,nfeat,right,nfeat,
     &  EW,work,lwork,info)

        write(6,*) "info=",info
        if(info.ne.0) then
        write(6,*) "diagonalization not successful"
        stop
        endif


       open(10,file="LPP_eigen_feat."//char(itype+48))
       rewind(10)
       do k=1,nfeat
       write(10,*) k,EW(k)
       enddo
       close(10)

       write(6,*) "EW", EW(1),EW(2),EW(3)

       allocate(PV(nfeat,numVR))
       allocate(PV0(nfeat0,numVR))

       PV0=0.d0

       
       ii=0
       do k=1,nfeat
       if(abs(EW(k)).gt.1.D-10.and.ii.lt.numVR) then
       ii=ii+1
       do j=1,nfeat
        PV(j,ii)=left(j,k)
        PV0(index(j),ii)=left(j,k)
       enddo
       endif
       enddo

       allocate(feat2_case(numVR,ncase))

       call dgemm('T','N',numVR,ncase,nfeat,1.d0,PV,nfeat,
     &  feat_case,nfeat,0.d0,feat2_case,numVR)


!       open(10,file="feat_LPP_stored0."//char(itype+48),
!     &  form="unformatted")
!       rewind(10) 
!       write(10) num_case(itype),numVR
!       do ii=1,num_case(itype)
!       write(10) ii,feat2_case(:,ii)
!       enddo
!       close(10)

       open(10,file="LPP."//char(itype+48))
       rewind(10) 
!       write(10) num_case(itype),numVR
       do ii=1,ncase
       write(10,"(10(E17.10,1x))") (feat2_case(j,ii),j=1,numVR)
       enddo
       close(10)


       open(10,file="eigv."//char(itype+48))
       rewind(10)
       write(10,*) nfeat0,numVR
       do j=1,nfeat0
       write(10,"(10(E17.7,1x))") (PV0(j,k),k=1,numVR)
       enddo
       close(10)


       deallocate(W_LPP)
       deallocate(D_LPP)
       deallocate(L_LPP)
       deallocate(left1)
       deallocate(left)

       deallocate(right1)
       deallocate(right)
       deallocate(work)
       deallocate(EW)

       deallocate(PV)

       deallocate(feat2_case)
2333   continue
!  In above, finish the new feature. We could have stopped here. 
!------------------------------------------------------------------
       stop
!cccccccccccccccccccccccccccccccccccccccccccccccccc

       end

       
