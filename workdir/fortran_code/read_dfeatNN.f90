  !// forquill v1.01 beta www.fcode.cn
module read_dfeatnn
    !implicit double precision (a-h, o-z)
    implicit none
  
  !!!!!!!!!!!!!          以下为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    integer(4) :: m_neigh                                  !模型所使用的最大近邻数(考虑这个数是否可以不用)
    integer(4) :: nfeat0m                                  !不同种原子的原始feature数目中最大者(目前似无意义)
 
    
    integer(4) :: natom                                    !image的原子个数  
     
    
    real*8,allocatable,dimension(:,:) :: force       !每个原子的受力
    real*8,allocatable,dimension(:,:,:,:) :: dfeat
    real*8,allocatable,dimension(:,:) :: feat    
    real*8, allocatable,dimension(:) :: energy        !每个原子的能量
    integer(4),allocatable,dimension(:,:) :: list_neigh
    integer(4),allocatable,dimension(:) :: iatom
  !!!!!!!!!!!!!          以上为  module variables     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
    contains

    subroutine deallo()
        deallocate(energy)
        deallocate(force)
        deallocate(feat)
        ! deallocate(num_neigh)
        deallocate(dfeat)
        deallocate(list_neigh)
        deallocate(iatom)
    end subroutine deallo
   
    subroutine read_dfeat(dfeatDir,image_Num,pos)
        integer(4)  :: image,nimage,nfeat0_tmp,jj,num_tmp,ii,i_p
        character(*),intent(in) :: dfeatDir
        integer(4), intent(in) :: image_Num
        integer(8), intent(in) :: pos
        real*8 AL(3,3)
        
        ! real*8,allocatable,dimension(:,:,:,:) :: dfeat
        ! real*8, allocatable,dimension(:,:) :: feat        
        real*8, allocatable, dimension (:,:) :: dfeat_tmp
        integer(4),allocatable, dimension (:) :: iat_tmp,jneigh_tmp,ifeat_tmp
        integer(4),allocatable,dimension(:) :: num_neigh    
        real*8,allocatable,dimension(:,:) :: xatom
        
        character(len=80) dfeatDirname

        dfeatDirname=trim(adjustl(dfeatDir))
        ! write(*,*) dfeatDirname
        ! open (23,file='/home/buyu/MLFF/AlHcomb_new/H2_hT/dfeat.fbin',action="read",form="unformatted",access='stream')
        open(23,file=trim(dfeatDirname),action="read",form="unformatted",access='stream')
        rewind(23)
        read(23) nimage,natom,nfeat0_tmp,m_neigh
        allocate(iatom(natom))
        read(23) iatom
        ! write(*,*) nimage,natom,nfeat0_tmp,m_neigh
        ! open(99,file='log.txt')         
        nfeat0m=nfeat0_tmp
         allocate(energy(natom))
         allocate(force(3,natom))
         allocate(feat(nfeat0m,natom))
         allocate(num_neigh(natom))
         allocate(list_neigh(m_neigh,natom))
         allocate(dfeat(nfeat0m,natom,m_neigh,3))
         allocate(xatom(3,natom))

    ! do 3000 image=1,image_Num
            ! write(*,*) image

        
     
        ! if (image.eq.image_Num) then
            read(23,pos=pos) energy
            
            read(23) force
            ! force = force*(-1)
            ! write(*,*) force(1,10)
            read(23) feat
            ! Inquire( 12 , Pos = i_p )
            ! write(*,*) i_p
            ! write(*,*) trim(dfeatDirname)
            ! write(*,*) image
            ! write(*,*) feat(19,1)
            read(23) num_neigh
            read(23) list_neigh
     !TODO:
            ! read(23) dfeat
            read(23) num_tmp
            ! write(*,*) num_tmp
            allocate(dfeat_tmp(3,num_tmp))
            allocate(iat_tmp(num_tmp))
            allocate(jneigh_tmp(num_tmp))
            allocate(ifeat_tmp(num_tmp))
            read(23) iat_tmp
            ! write(*,*) iat_tmp(3)
            read(23) jneigh_tmp
            read(23) ifeat_tmp
            read(23) dfeat_tmp
            
            read(23) xatom    ! xatom(3,natom)
            read(23) AL       ! AL(3,3)
            ! allocate(dfeat(nfeat0_tmp,natom,m_neigh,3))
            dfeat(:,:,:,:)=0.0
            do jj=1,num_tmp
            dfeat(ifeat_tmp(jj),iat_tmp(jj),jneigh_tmp(jj),:)=dfeat_tmp(:,jj)
            enddo
            deallocate(dfeat_tmp)
            deallocate(iat_tmp)
            deallocate(jneigh_tmp)
            deallocate(ifeat_tmp)
            close(23)
    !     else  
    !         read(23) energy
            
    !         read(23) force
    !         read(23) feat
    !         read(23) num_neigh
    !         read(23) list_neigh
    !  !TODO:
    !         ! read(23) dfeat
    !         read(23) num_tmp
    !         ! write(*,*) num_tmp
    !         allocate(dfeat_tmp(3,num_tmp))
    !         allocate(iat_tmp(num_tmp))
    !         allocate(jneigh_tmp(num_tmp))
    !         allocate(ifeat_tmp(num_tmp))
    !         read(23) iat_tmp
    !         ! write(*,*) iat_tmp(3)
    !         read(23) jneigh_tmp
    !         read(23) ifeat_tmp
    !         read(23) dfeat_tmp
            
    !         read(23) xatom    ! xatom(3,natom)
    !         read(23) AL       ! AL(3,3)
    !         deallocate(dfeat_tmp)
    !         deallocate(iat_tmp)
    !         deallocate(jneigh_tmp)
    !         deallocate(ifeat_tmp)
    !     end if

! 3000    continue
            
            
            do jj=1,natom
                ii=num_neigh(jj)+1
                list_neigh(ii:,jj)=0
                ! do ii=1,m_neigh
                !     if (abs(list_neigh(ii,jj)) .gt. natom) then
                !         list_neigh(ii,jj)=0
                !     end if
                ! end do
            enddo

        ! else
            ! read(23) 
    !         read(23)
    !         read(23) 
    !         read(23) 
    !         read(23) 
    !         read(23) 
    !  !TODO:
    !         ! read(23) dfeat
    !         read(23) 
        
    !         read(23) 
    !         read(23) 
    !         read(23) 
    !         read(23) 
            
    !         read(23) 
    !         read(23) 

    !     end if 
        
! 3000 continue
        
  

        ! deallocate(feat)
        ! deallocate(energy)
        !   deallocate(force)
        ! deallocate(feat)
        deallocate(num_neigh)
        ! deallocate(iatom)
        deallocate(xatom)



    end subroutine read_dfeat
  

end module read_dfeatnn
  
  
