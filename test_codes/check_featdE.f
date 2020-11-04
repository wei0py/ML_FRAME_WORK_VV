       program feat_LPP
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
       real*8,allocatable,dimension(:,:) :: storage
       integer,allocatable,dimension(:) :: index


       integer,allocatable,dimension(:) :: ipiv
       integer iatom_type(10)
       real*8 Ei_tmp,sum,epsilon,t_lpp
       integer itmp,lwork
       character(len=80),allocatable,dimension (:) :: trainSetFileDir
       character(len=80) trainSetDir,BadImageDir
       character(len=90) MOVEMENTDir,dfeatDir,infoDir,trainDataDir
       integer sys_num,sys
       

         write(6,*) "input itype"
         read(5,*) itype

         open(10,file="feat_new_stored."//char(itype+48),
     &    form="unformatted")
        rewind(10)
        read(10) ncase,nfeat
         
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

         write(6,*)  
     & "input: weight power mm(no weight,mm=0); Rcu_fact(~4000)"
         read(5,*) mm,fact

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


       sum=0.d0
       num=0
       do i1=1,ncase,50
       do i2=1,i1,50
       do j=1,nfeat
          sum=sum+(feat_case(j,i1)-feat_case(j,i2))**2
       enddo
       num=num+1
       enddo
       enddo

       sum=sum/num    ! this is the average distance
       dd_cut=sum/fact
       write(6,*) "dd_ave,dd_cut=",sum,dd_cut

       nstore=1000000
       allocate(storage(2,nstore))

       write(6,*) "big do loop"
       ncount=0
       do i1=1,ncase,10
       do i2=1,i1,10 
       sum=0.d0
       do j=1,nfeat
            sum=sum+(feat_case(j,i1)-feat_case(j,i2))**2
       enddo
         if(sum.lt.dd_cut) then
         ncount=ncount+1
         if(ncount.gt.nstore) then
           write(6,*) "ncount.gt.nstore, increase fact",ncount
           stop
         endif
           storage(1,ncount)=abs(Ei_case(i1)-Ei_case(i2))
           storage(2,ncount)=sum
         endif
       enddo
       enddo
       write(6,*) "finish: ncount=",ncount

       open(12,file="check.out")
       rewind(12)
       do i=1,ncount
       write(12,"(2(E14.7,1x))") storage(2,i),storage(1,i)
       enddo
       close(12)
      
!------------------------------------------------------------------
       stop
!cccccccccccccccccccccccccccccccccccccccccccccccccc

       end

       
