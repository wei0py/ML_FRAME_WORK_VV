PROGRAM classify_test
    use mod_bond_angle
    IMPLICIT NONE
    INTEGER :: ierr
    integer :: move_file=1101
    real*8 AL(3,3),Etotp
    real*8,allocatable,dimension (:,:) :: xatom,fatom
    real*8,allocatable,dimension (:) :: Eatom
    integer,allocatable,dimension (:) :: iatom,iatom_cluster
    logical nextline
    character(len=200) :: the_line
    integer num_step, natom, i, j,AtomIndex
    integer num_step0,num_step1,natom0,max_neigh
    real*8 Etotp_ave,E_tolerance
    real*8 sum1,sum2,sum3,sum4
    character*50 char_tmp(20)
    integer num_test(0:200)
    integer jj

    integer,allocatable,dimension (:,:,:) :: list_neigh,iat_neigh
    integer,allocatable,dimension (:,:) :: num_neigh
    real*8,allocatable,dimension (:,:,:,:) :: dR_neigh

    real*8,allocatable,dimension (:,:) :: feat
    real*8,allocatable,dimension (:,:,:,:) :: dfeat

    integer,allocatable,dimension (:,:) :: list_neigh_alltype
    integer,allocatable,dimension (:) :: num_neigh_alltype
    real*8,allocatable,dimension (:,:) :: feat_all
    real*8,allocatable,dimension (:,:) :: featk
    real*8,allocatable,dimension (:,:) :: bond_dist
    real*8,allocatable,dimension (:,:,:) :: theta_dist

    integer m_neigh,num,itype1,itype2,itype
    
    integer ntype,n2b,n3b1,n3b2,nfeat0
    integer iat,nbond,nangle,iat_class
    integer iat_type(100)
    real*8 Rc, Rc2,Rm
    integer iat_count

    real*8, allocatable, dimension (:,:) :: dfeat_tmp
    integer,allocatable, dimension (:) :: iat_tmp,jneigh_tmp,ifeat_tmp
    integer ii_tmp,jj_tmp,iat2,num_tmp
    real*8, allocatable, dimension (:,:,:,:,:,:,:,:) :: prob
    integer ii
    integer iflag_run

    real*8 value1_ave(10),value11_ave(10)
    integer num1_ave(10)

    integer nk_8(8)
    real*8 prob_dist(100,8)
    integer i1,i2,i3,i4,i5,i6,i7,i8
    integer j1,j2,j3,j4,j5,j6,j7,j8
    integer k1,k2,k3,k4,k5,k6,k7,k8
    integer ndim,nk_max

!    integer itype_bond(10),nxp_bond(10),nk_bond(10)
!    real*8  Rc_bond(10),Rg_bond(10),dRg_bond(10),XX_bond(10),scale_bond(10)
!    integer itype1_angle(10),itype2_angle(10),nxp_angle(10),nk_angle(10)
!    real*8 theta_angle(10),dtheta_angle(10),scale_angle(10),Rc_angle(10)

    integer nkk_8(10)
    integer ind_cluster(10,200), num_cluster
    real*8  feat_cluster(10,200)
    real*8 time
    character(len=80),allocatable,dimension (:) :: trainSetFileDir
    character(len=80) trainSetDir,BadImageDir
    character(len=90) MOVEMENTDir,dfeatDir,infoDir,trainDataDir,MOVEMENTallDir
    integer sys_num,sys,BadImageNum
    integer,allocatable,dimension (:) :: badimage

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    INTERFACE
        SUBROUTINE scan_title (io_file, title, title_line, if_find)
            CHARACTER(LEN=200), OPTIONAL :: title_line
            LOGICAL, OPTIONAL :: if_find
            INTEGER :: io_file
            CHARACTER(LEN=*) :: title
        END SUBROUTINE scan_title
    END INTERFACE
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    open(13,file="location")
    rewind(13)
    read(13,*) sys_num  !,trainSetDir
    read(13,'(a80)') trainSetDir
    allocate(trainSetFileDir(sys_num))
    do i=1,sys_num
    read(13,'(a80)') trainSetFileDir(i)    
    enddo
    close(13)
    MOVEMENTallDir=trim(trainSetDir)//"/MOVEMENTall"
    ! trainDataDir=trim(trainSetDir)//"/trainData.txt"
    BadImageDir=trim(trainSetDir)//"/imagesNotUsed"

    open(222,file=BadImageDir)
    rewind(222)
    read(222,*) BadImageNum
    if(BadImageNum.ne.0) then
    allocate(badimage(BadImageNum))
    do i=1,BadImageNum
        read(222,*) badimage(i)
    enddo
    endif
    close(222)

    open(10,file="classify.in",status="old",action="read")
    rewind(10)
    read(10,*) ntype
    do i=1,ntype
    read(10,*) iat_type(i)
    enddo
    read(10,*) Rc,m_neigh
    read(10,*) nbond
    do i=1,nbond
    read(10,*) itype_bond(i),Rc_bond(i),nxp_bond(i),Rg_bond(i),dRg_bond(i),XX_bond(i),scale_bond(i),nk_bond(i)
    enddo
    read(10,*) nangle 
    do i=1,nangle
    read(10,*)    &
    itype1_angle(i),itype2_angle(i),Rc_angle(i),theta_angle(i),dtheta_angle(i),nxp_angle(i),scale_angle(i),nk_angle(i)
    enddo
    close(10)

    write(6,*) "ntype=", ntype, "; input itype to classify"
    read(5,*) itype
    iat_class=iat_type(itype)

    allocate(bond_dist(1000,ntype))
    allocate(theta_dist(1000,ntype,ntype))

    ndim=nbond+nangle
    if(ndim.gt.8) then
    write(6,*) "ndim.gt.8, reduce the dimension", ndim
    stop
    endif

    nk_8=1

    num=0
    do i=1,nbond
    num=num+1
    nk_8(num)=nk_bond(i)
    enddo
    do i=1,nangle
    num=num+1
    nk_8(num)=nk_angle(i)
    enddo

    allocate(prob(nk_8(1),nk_8(2),nk_8(3),nk_8(4),nk_8(5),nk_8(6),nk_8(7),nk_8(8)))

!cccccccccccccccccccccccccccccccccccccccc
    value1_ave=0.d0
    value11_ave=0.d0
    num1_ave=0

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc    !
    iflag_run=0

9999  continue
    iflag_run=iflag_run+1
     if(iflag_run.eq.2) then
     allocate(feat_all(10,AtomIndex))
     endif
    prob=0.d0
    prob_dist=0.d0
    OPEN (move_file,file=MOVEMENTallDir,status="old",action="read") 
    rewind(move_file)

    if(iflag_run.eq.3) then
    open(33,file="MOVEMENT.cluster") 
    rewind(33)
    open(44,file='cluster_index.'//char(itype+48))
    rewind(44)
    endif


    max_neigh=-1
    num_step=0
    AtomIndex=0
    iat_count=0
    num_test=0
    bond_dist=0.d0
    theta_dist=0.d0
1000  continue

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    call scan_title (move_file,"ITERATION",if_find=nextline)

    if(.not.nextline) goto 2000
    num_step=num_step+1
    if(BadImageNum.ne.0) then 
    do i=1,BadImageNum
        if (num_step.eq.badimage(i)) then
        goto 1000
        endif
    enddo
    endif
    

    backspace(move_file) 
    read(move_file, *) natom,char_tmp(1:4),time
    ALLOCATE (iatom(natom),xatom(3,natom),fatom(3,natom),Eatom(natom))
    allocate (iatom_cluster(natom))
    iatom_cluster=0

        CALL scan_title (move_file, "LATTICE")
        DO j = 1, 3
            READ (move_file,*) AL(1:3,j)
        ENDDO

       CALL scan_title (move_file, "POSITION")
        DO j = 1, natom
            READ(move_file, *) iatom(j),xatom(1,j),xatom(2,j),xatom(3,j)
        ENDDO

        CALL scan_title (move_file, "FORCE", if_find=nextline)
        if(.not.nextline) then
          write(6,*) "force not found, stop", num_step
          stop
        endif
        DO j = 1, natom
            READ(move_file, *) iatom(j),fatom(1,j),fatom(2,j),fatom(3,j)
        ENDDO

    CALL scan_title (move_file, "ATOMIC-ENERGY",if_find=nextline)
       if(.not.nextline) then
         write(6,*) "Atomic-energy not found, stop",num_step
         stop
        endif

        backspace(move_file)
        read(move_file,*) char_tmp(1:4),Etotp

        DO j = 1, natom
            READ(move_file, *) iatom(j),Eatom(j)
        ENDDO

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

    allocate(list_neigh(m_neigh,ntype,natom))
    allocate(iat_neigh(m_neigh,ntype,natom))
    allocate(dR_neigh(3,m_neigh,ntype,natom))   ! d(neighbore)-d(center) in xyz
    allocate(num_neigh(ntype,natom))

    call find_neighbore(iatom,natom,xatom,AL,Rc,num_neigh,list_neigh, &
       dR_neigh,iat_neigh,ntype,iat_type,m_neigh)


!ccccccccccccccccccccccccccccccccc

     allocate(featk(10,natom))

    call classify3(natom,Rc,nk_8,num_neigh, &
       list_neigh,dR_neigh,iat_neigh,ntype,m_neigh,iatom,iat_type, &
       iat_class,nbond,nangle, &
       value1_ave,value11_ave,num1_ave,   &
       prob,prob_dist,iflag_run,num_cluster,ind_cluster,&
       iatom_cluster,featk,feat_cluster,bond_dist,theta_dist)

      if(iflag_run.eq.2) then
      do j=1,natom
      if(iatom(j).eq.iat_class) then
      iat_count=iat_count+1
      feat_all(:,iat_count)=featk(:,j)
      endif
      enddo
      endif
      deallocate(featk)

     

     if(iflag_run.eq.3) then
     write(33,*) natom,"atoms, Iteration fs)= ", time
     write(33,*) "Lattice vector (Angstrom)"
     write(33,"(3(E17.10,2x))") AL(:,1)
     write(33,"(3(E17.10,2x))") AL(:,2)
     write(33,"(3(E17.10,2x))") AL(:,3)
     write(33,*) "Position (normalized)"
     do iat=1,natom
!     write(33,"(i4,3(f17.12,2x),' 1 1 1  ', i4,2x,i10)") iatom(iat),xatom(1,iat), &
     write(33,"(i4,3(f17.12,2x),' 1 1 1  ', i4,2x,i10)") iatom(iat)+iatom_cluster(iat),xatom(1,iat), &
       xatom(2,iat),xatom(3,iat), iatom_cluster(iat), int(AtomIndex+iat)
     write(44,"(i5,2x,i4,2x,i10)") iatom(iat),iatom_cluster(iat),int(AtomIndex+iat)

     num_test(iatom_cluster(iat))=num_test(iatom_cluster(iat))+1

     enddo
     write(33,*) "------------------------------------"
     
     
     endif

     AtomIndex=AtomIndex+natom
!cccccccccccccccccccccccccccccccccccccccccccccchhhhhh
!cccccccccccccccccccccccccccccccccccccccccccccchhhhhh
    deallocate(list_neigh)
    deallocate(iat_neigh)
    deallocate(dR_neigh)
    deallocate(num_neigh)


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      DEALLOCATE (iatom,xatom,fatom,Eatom)
      deallocate (iatom_cluster)
!--------------------------------------------------------
       goto 1000     
2000   continue    
      close(move_file)


      if(iflag_run.eq.1) then
      open(13,file="bond_dist."//char(itype+48)) 
      rewind(13)
      write(13,"('#',20(i4,1x))") (iat_type(itype1),itype1=1,ntype)
      do ii=1,1000
      write(13,"(f12.6,2x,20(E11.4,1x))") Rc/998.d0*(ii-1),(bond_dist(ii,itype1),itype1=1,ntype)
      enddo
      close(13)

      open(13,file="theta_dist."//char(itype+48)) 
      rewind(13)
      write(13,"('#',20(i8,1x))") ((iat_type(itype1)*10000+iat_type(itype2),itype2=1,itype1),itype1=1,ntype)
      do ii=1,1000
      write(13,"(f12.6,2x,20(E11.4,1x))") 180/998.d0*(ii-1),  &
       ((theta_dist(ii,itype1,itype2),itype2=1,itype1),itype1=1,ntype)
      enddo
      close(13)
      endif

      if(iflag_run.eq.3) then
      close(33)
      close(44)

!        do ii=1,num_cluster
!        write(6,*) "ncount=", ii,num_test(ii)
!        enddo

       stop
      endif


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if(iflag_run.eq.1) then
      do ii=1,ndim
      value1_ave(ii)=value1_ave(ii)/num1_ave(ii)
      value11_ave(ii)=value11_ave(ii)/num1_ave(ii)
      value11_ave(ii)=nk_8(ii)/4.d0/(dsqrt(abs(value11_ave(ii)-value1_ave(ii)**2))+0.000001)
      write(6,*) "value_ave", value1_ave(ii),value11_ave(ii)
      enddo

      goto 9999
      endif

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      do ii=1,8
      if(nk_8(ii).eq.1) then
      nkk_8(ii)=0
      else
      nkk_8(ii)=1
      endif
      enddo


     num_cluster=0
     do i8=1,nk_8(8)
     do i7=1,nk_8(7)
     do i6=1,nk_8(6)
     do i5=1,nk_8(5)
     do i4=1,nk_8(4)
     do i3=1,nk_8(3)
     do i2=1,nk_8(2)
     do i1=1,nk_8(1)

       if(prob(i1,i2,i3,i4,i5,i6,i7,i8).eq.0.d0) goto 3002

         do j8=-nkk_8(8),nkk_8(8)
         k8=i8+j8
         if(k8.lt.1) k8=1
         if(k8.gt.nk_8(8)) k8=nk_8(8)

         do j7=-nkk_8(7),nkk_8(7)
         k7=i7+j7
         if(k7.lt.1) k7=1
         if(k7.gt.nk_8(7)) k7=nk_8(7)

         do j6=-nkk_8(6),nkk_8(6)
         k6=i6+j6
         if(k6.lt.1) k6=1
         if(k6.gt.nk_8(6)) k6=nk_8(6)

         do j5=-nkk_8(5),nkk_8(5)
         k5=i5+j5
         if(k5.lt.1) k5=1
         if(k5.gt.nk_8(5)) k5=nk_8(5)

         do j4=-nkk_8(4),nkk_8(4)
         k4=i4+j4
         if(k4.lt.1) k4=1
         if(k4.gt.nk_8(4)) k4=nk_8(4)

         do j3=-nkk_8(3),nkk_8(3)
         k3=i3+j3
         if(k3.lt.1) k3=1
         if(k3.gt.nk_8(3)) k3=nk_8(3)

         do j2=-nkk_8(2),nkk_8(2)
         k2=i2+j2
         if(k2.lt.1) k2=1
         if(k2.gt.nk_8(2)) k2=nk_8(2)

         do j1=-nkk_8(1),nkk_8(1)
         k1=i1+j1
         if(k1.lt.1) k1=1
         if(k1.gt.nk_8(1)) k1=nk_8(1)


       if(prob(k1,k2,k3,k4,k5,k6,k7,k8).gt. &
          prob(i1,i2,i3,i4,i5,i6,i7,i8)+0.001.and. &
        abs(j1)+abs(j2)+abs(j3)+abs(j4)+abs(j5)+abs(j6)+abs(j7)+abs(j8).ne.0 ) goto 3002

         enddo
         enddo
         enddo
         enddo
         enddo
         enddo
         enddo
         enddo


       num_cluster=num_cluster+1
       write(6,"('cluster ',i4,'  indx=', 8(i2,1x),2x,E12.5)") &
        num_cluster,  i1,i2,i3,i4,i5,i6,i7,i8,prob(i1,i2,i3,i4,i5,i6,i7,i8)
       if(num_cluster.gt.200) then
         write(6,*) "num_cluster.gt.200,stop"
         stop
       endif

       ind_cluster(1,num_cluster)=i1
       ind_cluster(2,num_cluster)=i2
       ind_cluster(3,num_cluster)=i3
       ind_cluster(4,num_cluster)=i4
       ind_cluster(5,num_cluster)=i5
       ind_cluster(6,num_cluster)=i6
       ind_cluster(7,num_cluster)=i7
       ind_cluster(8,num_cluster)=i8

3002   continue

       enddo
       enddo
       enddo
       enddo
       enddo
       enddo
       enddo
       enddo

!cccccccccccccccccccccccccccccccccccccccccccc
     nk_max=0
     do k1=1,8
     if(nk_8(k1).gt.nk_max) nk_max=nk_8(k1)
     enddo

      if(iflag_run.eq.2) then

     open(10,file="feat_dist."//char(itype+48)) 
     rewind(10)
     do ii=1,nk_max
     write(10,"(i4,8(E12.5,1x))") ii, (prob_dist(ii,k1),k1=1,ndim)
     enddo
     close(10)

      write(6,*) "num_cluster=",num_cluster
      call kmean(feat_all,iat_count,ind_cluster,feat_cluster,num_cluster)

      open(13,file="classf_all."//char(itype+48))
      rewind(13)
      write(13,*) iat_count,ndim
      do ii=1,iat_count
      write(13,"(8(E14.7,1x))") (feat_all(jj,ii),jj=1,ndim)
      enddo
      close(13)

      open(14,file="classf_cluster."//char(itype+48))
      rewind(14)
      write(14,*) num_cluster,ndim
      do ii=1,num_cluster
      write(14,"(8(E14.7,1x))") (feat_cluster(jj,ii),jj=1,ndim)
      enddo
      close(14)


      endif

     if(iflag_run.eq.2) goto 9999

       stop
       end
