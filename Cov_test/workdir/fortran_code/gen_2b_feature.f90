module calc_ftype1

    IMPLICIT NONE
    INTEGER :: ierr
    integer,allocatable,dimension (:) :: iatom
    integer num_step, natom, i, j
    integer num_step0,num_step1,natom0,max_neigh
    real*8 Etotp_ave,E_tolerance

    integer sys_num,sys,recalc_grid


    real*8,allocatable,dimension (:,:) :: grid2
    real*8,allocatable,dimension (:,:,:) :: grid2_2
    integer n2b_t,n3b1_t,n3b2_t,it
    integer n2b_type(100),n2bm

    real*8 Rc_M
    integer n3b1m,n3b2m,kkk,ii

    real*8,allocatable,dimension (:,:) :: feat
    real*8,allocatable,dimension (:,:,:,:) :: dfeat

    integer,allocatable,dimension (:,:) :: list_neigh_alltypeM
    integer,allocatable,dimension (:) :: num_neigh_alltypeM

    real*8 sum1,diff

    integer m_neigh,num,itype1,itype2,itype
    integer iat1,max_neigh_M,num_M
    
    integer ntype,n2b,n3b1,n3b2,nfeat0m,nfeat0(100)
    integer ntype1,ntype2,k1,k2,k12,ii_f,iat,ixyz
    integer iat_type(100)
    real*8 Rc, Rc2,Rm
    real*8 alpha31,alpha32

    ! real*8, allocatable, dimension (:,:) :: dfeat_tmp
    ! integer,allocatable, dimension (:) :: iat_tmp,jneigh_tmp,ifeat_tmp
    integer ii_tmp,jj_tmp,iat2,num_tmp,num_tot,i_tmp,jjj,jj
    integer iflag_grid,iflag_ftype
    real*8 fact_grid,dR_grid1,dR_grid2

    real*8 Rc_type(100), Rc2_type(100), Rm_type(100),fact_grid_type(100),dR_grid1_type(100),dR_grid2_type(100)
    integer iflag_grid_type(100),n3b1_type(100),n3b2_type(100)


!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

contains
    subroutine load_model()
    
        open(10,file="input/gen_2b_feature.in",status="old",action="read")
        rewind(10)
        read(10,*) Rc_M,m_neigh
        read(10,*) ntype
        do i=1,ntype
        read(10,*) iat_type(i)
        read(10,*) Rc_type(i),Rm_type(i),iflag_grid_type(i),fact_grid_type(i),dR_grid1_type(i)
        read(10,*) n2b_type(i)
    
         if(Rc_type(i).gt.Rc_M) then
          write(6,*) "Rc_type must be smaller than Rc_M, gen_3b_feature.in",i,Rc_type(i),Rc_M
          stop
         endif
    
        enddo
        read(10,*) E_tolerance
        read(10,*) iflag_ftype
        read(10,*) recalc_grid
        close(10)

    !cccccccccccccccccccccccccccccccccccccccc
    
        do i=1,ntype
        if(iflag_ftype.eq.3.and.iflag_grid_type(i).ne.3) then
        write(6,*) "if iflag_ftype.eq.3, iflag_grid must equal 3, stop"
        stop
        endif
        enddo
    
         n2bm=0
         do i=1,ntype
         if(n2b_type(i).gt.n2bm) n2bm=n2b_type(i)
         enddo
    
    !cccccccccccccccccccccccccccccccccccccccccccccccc
         nfeat0m=ntype*n2bm
         write(6,*) "max,nfeat0m=",nfeat0m
    
         do itype=1,ntype
         nfeat0(itype)=n2b_type(itype)*ntype
         enddo
         write(6,*) "itype,nfeat0=",(nfeat0(itype),itype=1,ntype)
    
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
    
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
        allocate(grid2(0:n2bm+1,ntype))
        allocate(grid2_2(2,n2bm+1,ntype))
    !cccccccccccccccccccccccccccccccccccccccccccccccccccc
         do kkk=1,ntype    ! center atom
         
         Rc=Rc_type(kkk)
         Rm=Rm_type(kkk)
         iflag_grid=iflag_grid_type(kkk)
         fact_grid=fact_grid_type(kkk)
         dR_grid1=dR_grid1_type(kkk)
         n2b=n2b_type(kkk)
    
    !cccccccccccccccccccccccccccccccccccccccc
    
        if(iflag_grid.eq.1.or.iflag_grid.eq.2) then
    
            open(10,file="output/grid2b_type12."//char(kkk+48))
            rewind(10)
            do i=0,n2b+1
            read(10,*) grid2(i,kkk)
            read(10,*) grid2(i,kkk)
            read(10,*) grid2(i,kkk)
            enddo
            close(10)
    
         endif   ! iflag_grid.eq.1,2
    
    !cccccccccccccccccccccccccccccccccccccccccccc
        if(iflag_grid.eq.3) then  
     ! for iflag_grid.eq.3, the graid is just read in. 
     ! Its format is different from above grid31, grid32. 
     ! For each point, it just have two numbers, r1,r2, indicating the region of the sin peak function.
    
        open(13,file="output/grid2b_type3."//char(kkk+48))
        rewind(13)
        read(13,*) n2b_t
        if(n2b_t.ne.n2b) then
        write(6,*) "n2b_t.ne.n2b,in grid2b_type3", n2b_t,n2b
        stop
        endif
        do i=1,n2b
        read(13,*) it,grid2_2(1,i,kkk),grid2_2(2,i,kkk)
        if(grid2_2(2,i,kkk).gt.Rc_type(kkk)) write(6,*) "grid2_2.gt.Rc",grid2_2(2,i,kkk),Rc_type(kkk)
        enddo
        close(13)
        endif
    
        enddo     ! kkk=1,ntype

!cccccccccccccccccccccccccccccccccccccccccccccccccccc
!  FInish the initial grid treatment

    end subroutine load_model
    
    subroutine set_image_info(atom_type_list,is_reset)
        integer(4),dimension(:),intent(in) :: atom_type_list
        logical,intent(in) :: is_reset
        integer(4) :: image_size
        
        image_size=size(atom_type_list)
        if (is_reset .or. (.not. allocated(iatom)) .or. image_size/=natom) then
        
            if (allocated(iatom))then
                if (image_size==natom .and. maxval(abs(atom_type_list-iatom))==0) then
                    return
                end if
                deallocate(iatom)
            end if
              
            natom=image_size
            allocate(iatom(natom))                   
            iatom=atom_type_list
              
        end if
        
    end subroutine set_image_info
    

subroutine gen_feature(AL,xatom)
    integer(4)  :: itype,i,j

    real(8), intent(in) :: AL(3,3)
    real(8),dimension(:,:),intent(in) :: xatom

    integer,allocatable,dimension (:,:,:) :: list_neigh,iat_neigh,iat_neigh_M
    integer,allocatable,dimension (:,:) :: num_neigh
    real*8,allocatable,dimension (:,:,:,:) :: dR_neigh
    integer,allocatable,dimension (:,:) :: map2neigh_alltypeM
    integer,allocatable,dimension (:,:) :: list_tmp
    integer,allocatable,dimension (:) :: nfeat_atom
    integer,allocatable,dimension (:) :: itype_atom
    integer,allocatable,dimension (:,:,:) :: map2neigh_M
    integer,allocatable,dimension (:,:,:) :: list_neigh_M
    integer,allocatable,dimension (:,:) :: num_neigh_M
    integer,allocatable,dimension (:,:) :: list_neigh_alltype
    integer,allocatable,dimension (:) :: num_neigh_alltype
    ! real*8,allocatable,dimension (:,:,:,:) :: dfeat
!cccccccccccccccccccccccccccccccccccccccccccccccccccc
    if (allocated(dfeat)) then
        
        deallocate(feat)
        deallocate(dfeat)
        deallocate(list_neigh_alltypeM)
        deallocate(num_neigh_alltypeM)
    endif

    allocate(list_neigh(m_neigh,ntype,natom))
    allocate(map2neigh_M(m_neigh,ntype,natom)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
    allocate(list_neigh_M(m_neigh,ntype,natom)) ! the neigh list of Rc_M
    allocate(num_neigh_M(ntype,natom))
    allocate(iat_neigh(m_neigh,ntype,natom))
    allocate(dR_neigh(3,m_neigh,ntype,natom))   ! d(neighbore)-d(center) in xyz
    allocate(num_neigh(ntype,natom))
    allocate(list_neigh_alltype(m_neigh,natom))
    allocate(num_neigh_alltype(natom))

    allocate(iat_neigh_M(m_neigh,ntype,natom))
    allocate(list_neigh_alltypeM(m_neigh,natom))
    allocate(num_neigh_alltypeM(natom))
    allocate(map2neigh_alltypeM(m_neigh,natom)) ! from list_neigh(of this feature) to list_neigh_all (of Rc_M
    allocate(list_tmp(m_neigh,ntype))
    allocate(itype_atom(natom))
    allocate(nfeat_atom(natom))

    allocate(feat(nfeat0m,natom))
    allocate(dfeat(nfeat0m,natom,m_neigh,3))

!ccccccccccccccccccccccccccccccccccccccccccc
    itype_atom=0
    do i=1,natom
    do j=1,ntype
     if(iatom(i).eq.iat_type(j)) then
      itype_atom(i)=j
     endif
    enddo
      if(itype_atom(i).eq.0) then
      write(6,*) "this atom type didn't found", itype_atom(i)
      stop
      endif
     enddo
!ccccccccccccccccccccccccccccccccccccccccccc

    call find_neighbore(iatom,natom,xatom,AL,Rc_type,num_neigh,list_neigh, &
       dR_neigh,iat_neigh,ntype,iat_type,m_neigh,Rc_M,map2neigh_M,list_neigh_M, &
       num_neigh_M,iat_neigh_M)

!ccccccccccccccccccccccccccccccccc
!ccccccccccccccccccccccccccccccccc

      max_neigh=-1
      num_neigh_alltype=0
      max_neigh_M=-1
      num_neigh_alltypeM=0
      list_neigh_alltypeM=0

      do iat=1,natom
      list_neigh_alltype(1,iat)=iat
      list_neigh_alltypeM(1,iat)=iat


      num_M=1
      do itype=1,ntype
      do j=1,num_neigh_M(itype,iat)
      num_M=num_M+1
      if(num_M.gt.m_neigh) then
      write(6,*) "total num_neigh.gt.m_neigh,stop",m_neigh
      stop
      endif
      list_neigh_alltypeM(num_M,iat)=list_neigh_M(j,itype,iat)
      list_tmp(j,itype)=num_M
      enddo
      enddo


      num=1
      map2neigh_alltypeM(1,iat)=1
      do  itype=1,ntype
      do   j=1,num_neigh(itype,iat)
      num=num+1
      list_neigh_alltype(num,iat)=list_neigh(j,itype,iat)
      map2neigh_alltypeM(num,iat)=list_tmp(map2neigh_M(j,itype,iat),itype)
! map2neigh_M(j,itype,iat), maps the jth neigh in list_neigh(Rc) to jth' neigh in list_neigh_M(Rc_M) 
      enddo
      enddo

!ccccccccccccccccccccccccccccccccccccccc


      num_neigh_alltype(iat)=num
      num_neigh_alltypeM(iat)=num_M
      if(num.gt.max_neigh) max_neigh=num
      if(num_M.gt.max_neigh_M) max_neigh_M=num_M
      enddo  ! iat

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! This num_neigh_alltype(iat) include itself !
    dfeat=0.d0
    feat=0.d0
    if(iflag_ftype.eq.1) then
! iflag_ftype.eq.1, the sin peak span over two grid points
    call find_feature_2b_type1(natom,itype_atom,Rc_type,n2b_type,num_neigh,  &
       list_neigh,dR_neigh,iat_neigh,ntype,grid2, &
       feat,dfeat,nfeat0m,m_neigh,n2bm,nfeat_atom)
    endif
    if(iflag_ftype.eq.2) then
!  iflag_ftype.eq.2, the sin peak span over three grid points
    call find_feature_2b_type2(natom,itype_atom,Rc_type,n2b_type,num_neigh,  &
       list_neigh,dR_neigh,iat_neigh,ntype,grid2, &
       feat,dfeat,nfeat0m,m_neigh,n2bm,nfeat_atom)
    endif
    if(iflag_ftype.eq.3) then
!  iflag_ftype.eq.3, the sin peak span over the two ends specified by grid31_2,grid32_2
!  So, there could be many overlaps between different sin peaks
    call find_feature_2b_type3(natom,itype_atom,Rc_type,n2b_type,num_neigh,  &
       list_neigh,dR_neigh,iat_neigh,ntype,grid2_2, &
       feat,dfeat,nfeat0m,m_neigh,n2bm,nfeat_atom)
    endif
!cccccccccccccccccccccccccccccccccccccccccccccccccccc


    deallocate(list_neigh)
    deallocate(iat_neigh)
    deallocate(dR_neigh)
    deallocate(num_neigh)
    ! deallocate(feat)
    ! deallocate(dfeat)
    deallocate(list_neigh_alltype)
    deallocate(num_neigh_alltype)


    deallocate(list_neigh_M)
    deallocate(num_neigh_M)
    deallocate(map2neigh_M)
    deallocate(iat_neigh_M)
    ! deallocate(list_neigh_alltypeM)
    ! deallocate(num_neigh_alltypeM)
    deallocate(map2neigh_alltypeM)
    deallocate(list_tmp)
    deallocate(itype_atom)
    deallocate(nfeat_atom)

!--------------------------------------------------------


end subroutine gen_feature

end module calc_ftype1