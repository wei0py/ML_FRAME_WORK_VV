PROGRAM kmean_RD
    IMPLICIT double precision (a-h,o-z)
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
    integer iatom_type(20)
    integer ncount_cluster(200)


    real*8,allocatable,dimension (:,:) :: feat_case
    real*8,allocatable,dimension (:) :: Ei_case,weight_feat
    real*8,allocatable,dimension (:,:) :: PV

    real*8,allocatable,dimension (:,:) :: feat_all,feat_center,feat_cluster
    integer,allocatable,dimension (:) :: id_cluster

   

    integer m_neigh,num,itype1,itype2,itype
    
    integer ntype,n2b,n3b1,n3b2,nfeat0
    integer iat,nbond,nangle,iat_class
    integer iat_type(100)
    real*8 Rc, Rc2,Rm
    integer iat_count

    integer ii_tmp,jj_tmp,iat2,num_tmp
    integer ii
    integer iflag_run


    integer ndim,nk_max

!    integer itype_bond(10),nxp_bond(10),nk_bond(10)
!    real*8  Rc_bond(10),Rg_bond(10),dRg_bond(10),XX_bond(10),scale_bond(10)
!    integer itype1_angle(10),itype2_angle(10),nxp_angle(10),nk_angle(10)
!    real*8 theta_angle(10),dtheta_angle(10),scale_angle(10),Rc_angle(10)

    integer nkk_8(10)
    integer ind_cluster(10,200), num_cluster
    real*8 time
    character*50 iat_class_char
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

       open(10,file="LPP.input") 
       rewind(10)
       read(10,*) ntype,numVR
       do i=1,ntype
          read(10,*)iatom_type(i)
       end do
       close(10)


         write(6,*) "ntype=", ntype, " input itype"
         read(5,*) itype
         write(6,*) "input num of cluster,iseed(negative)"
         read(5,*) num_cluster,iseed


         open(10,file="feat_new_stored."//char(itype+48), &
         form="unformatted")
        rewind(10)
        read(10) ncase_tmp,nfeat
         write(6,*) "ncase=", ncase_tmp
         
         ninterv=1

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

        open(10,file="eigv."//char(itype+48))
        rewind(10)
        read(10,*) nfeat0,numVR
        if(nfeat0.ne.nfeat) then
        write(6,*) "nfeat0.ne.nfeat,stop",nfeat0,nfeat
        stop
        endif
        allocate(PV(nfeat,numVR))
        do j=1,nfeat
        read(10,*) (PV(j,k),k=1,numVR)
        enddo
        close(10)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
        allocate(feat_all(numVR,ncase))
        
        call dgemm('T','N',numVR,ncase,nfeat,1.d0,PV,nfeat, &
         feat_case,nfeat,0.d0,feat_all,numVR)

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! The following is the real kmean, with random selected num_cluster
       allocate(feat_center(numVR,num_cluster))
       allocate(id_cluster(ncase))
       allocate(feat_cluster(numVR,num_cluster))

       do jj=1,num_cluster
       xx=ran1(iseed)
       jj1=xx*(ncase-1)+1
       if(jj1.gt.ncase) jj1=ncase
       do i=1,numVR
       feat_center(i,jj)=feat_all(i,jj1)
       enddo
       enddo

       do 1001 iter=1,20

       feat_cluster=0.d0
       ncount_cluster=0

       do 100 icase=1,ncase
       dist_min=1.D+30
       do jj=1,num_cluster
       sum=0.d0
       do i=1,numVR
       sum=sum+(feat_all(i,icase)-feat_center(i,jj))**2
       enddo
       if(sum.lt.dist_min) then
       dist_min=sum
       jjmin=jj
       endif
       enddo
       id_cluster(icase)=jjmin
       feat_cluster(:,jjmin)=feat_cluster(:,jjmin)+feat_all(:,icase)
       ncount_cluster(jjmin)=ncount_cluster(jjmin)+1
100    continue

       diff=0.d0
       do jj=1,num_cluster
       do i=1,numVR
       feat_cluster(i,jj)=feat_cluster(i,jj)/ncount_cluster(jj)
       diff=diff+(feat_cluster(i,jj)-feat_center(i,jj))**2
       feat_center(i,jj)=feat_cluster(i,jj)
       enddo
       enddo
       write(6,*) "feature cluster diff",iter,diff
1001   continue

       do jj=1,num_cluster
       write(6,*) "ncount,cluster", ncount_cluster(jj)
       enddo

       open(10,file="classf_cluster."//char(itype+48))
       rewind(10)
       write(10,*) num_cluster,numVR
       do jj=1,num_cluster
       write(10,"(10(E14.7,1x))") (feat_center(i,jj),i=1,numVR)
       enddo
       close(10)

       open(12,file="classf_all."//char(itype+48))
       rewind(12)
       write(12,*) ncase,numVR
       do ii=1,ncase
       write(12,"(10(E14.7,1x))") (feat_all(i,ii),i=1,numVR)
       enddo
       close(12)
      


!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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


9999  continue
    prob=0.d0
    prob_dist=0.d0
    OPEN (move_file,file=MOVEMENTallDir,status="old",action="read") 
    rewind(move_file)

    open(33,file="MOVEMENT.cluster") 
    rewind(33)

    max_neigh=-1
    num_step=0
    AtomIndex=0
    iat_count=0
    num_test=0
    bond_dist=0.d0
    theta_dist=0.d0
    icase=0
    istep=0
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

    istep=istep+1
    
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
!cccccccccccccccccccccccc, very dangerous operation, require the match between feat_new_stored, and MOVEMENTall
        if(iatom(j).eq.iatom_type(itype)) then
        icase=icase+1
        if(icase.gt.ncase) then
        write(6,*) "icase,gt.ncase,stop",ncase,nstep
        stop
        endif
        iatom_cluster(j)=id_cluster(icase)
        else
        iatom_cluster(j)=0
        endif

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

     enddo
     write(33,*) "------------------------------------"
     

     AtomIndex=AtomIndex+natom
!cccccccccccccccccccccccccccccccccccccccccccccchhhhhh
!cccccccccccccccccccccccccccccccccccccccccccccchhhhhh
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      DEALLOCATE (iatom,xatom,fatom,Eatom)
      deallocate (iatom_cluster)
!--------------------------------------------------------
       goto 1000     
2000   continue    
      close(move_file)
      close(33)

      write(6,*) "icase,ncase",icase,ncase
      stop
      end
